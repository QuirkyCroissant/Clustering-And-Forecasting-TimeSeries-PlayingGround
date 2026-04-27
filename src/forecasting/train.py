from tqdm.auto import tqdm
import numpy as np

XGB_PARAMETER_PROFILES = {
    "baseline": dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
    ),
    "regularized": dict(
        n_estimators=700,
        learning_rate=0.035,
        max_depth=6,
        min_child_weight=5.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=3.0,
    ),
    "shallow": dict(
        n_estimators=900,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=10.0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=5.0,
    ),
    "deeper": dict(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=10,
        min_child_weight=3.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.02,
        reg_lambda=2.0,
    ),
}


def resolve_xgb_params(profile: str = "regularized", overrides: dict | None = None) -> dict:
    if profile not in XGB_PARAMETER_PROFILES:
        raise ValueError(f"Unknown xgb profile {profile!r}. Available: {sorted(XGB_PARAMETER_PROFILES)}")
    params = dict(XGB_PARAMETER_PROFILES[profile])
    if overrides:
        params.update(overrides)
    return params


def get_model(
    model_name: str,
    random_state: int = 42,
    use_gpu: bool = False,
    model_params: dict | None = None,
    xgb_profile: str = "regularized",
):
    model_name = model_name.lower()

    if model_name == "lgbm":
        from lightgbm import LGBMRegressor

        params = dict(
            objective="mae",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=1,
        )
        if use_gpu:
            params["device"] = "gpu"
        return LGBMRegressor(**params)

    if model_name == "xgb":
        from xgboost import XGBRegressor

        params = dict(
            objective="reg:absoluteerror",
            tree_method="hist",
            random_state=random_state,
            verbosity=1,
        )
        params.update(resolve_xgb_params(profile=xgb_profile, overrides=model_params))
        if use_gpu:
            params["device"] = "cuda"
        return XGBRegressor(**params)

    raise ValueError(f"Unsupported model_name: {model_name}")


def _split_train_valid(df):
    unique_ds = np.sort(df["ds"].unique())

    if len(unique_ds) < 2:
        raise ValueError("Need at least two unique ds values to create a validation split")

    split_idx = int(len(unique_ds) * 0.8)
    split_idx = min(max(split_idx, 1), len(unique_ds) - 1)
    return unique_ds[split_idx]


def _fit_model(model, model_name, X_train, y_train, X_valid, y_valid, verbose):
    if model_name == "xgb":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=verbose,
        )
        return model.evals_result()

    model.fit(X_train, y_train)
    return {}


def _build_fit_metadata(
    train_part,
    valid_part,
    split_ds,
    feature_cols,
    model_name,
    use_gpu,
    extra=None,
):
    metadata = {
        "model_name": model_name.lower(),
        "use_gpu": bool(use_gpu),
        "split_ds": str(split_ds),
        "n_features": len(feature_cols),
        "feature_cols": list(feature_cols),
        "n_rows_train": len(train_part),
        "n_rows_valid": len(valid_part),
        "n_households_train": train_part["ID"].nunique(),
        "n_households_valid": valid_part["ID"].nunique(),
    }
    if extra:
        metadata.update(extra)
    return metadata


def fit_global_model(
    train_df,
    feature_cols,
    model_name="lgbm",
    use_gpu=False,
    model_params=None,
    xgb_profile="regularized",
    verbose=50,
):
    model = get_model(
        model_name,
        use_gpu=use_gpu,
        model_params=model_params,
        xgb_profile=xgb_profile,
    )

    split_ds = _split_train_valid(train_df)
    train_mask = train_df["ds"] < split_ds
    valid_mask = ~train_mask

    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, "target"].to_numpy(copy=False)
    X_valid = train_df.loc[valid_mask, feature_cols]
    y_valid = train_df.loc[valid_mask, "target"].to_numpy(copy=False)

    eval_log = _fit_model(
        model=model,
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        verbose=verbose,
    )

    metadata = _build_fit_metadata(
        train_part=train_df.loc[train_mask, ["ID", "ds"]],
        valid_part=train_df.loc[valid_mask, ["ID", "ds"]],
        split_ds=split_ds,
        feature_cols=feature_cols,
        model_name=model_name,
        use_gpu=use_gpu,
        extra={
            "scope": "global",
            "xgb_profile": xgb_profile if model_name.lower() == "xgb" else None,
            "model_params": model_params or {},
        },
    )

    return {
        "model": model,
        "eval_log": eval_log,
        "metadata": metadata,
    }


def fit_cluster_models(
    train_df,
    feature_cols,
    model_name="lgbm",
    min_households=30,
    min_rows=500,
    use_gpu=False,
    model_params=None,
    xgb_profile="regularized",
    show_progress=True,
    verbose=False,
):
    model_dict = {}
    groups = train_df.groupby("ForecastGroup", observed=True)

    iterator = groups
    if show_progress:
        iterator = tqdm(
            groups,
            total=groups.ngroups,
            desc=f"Training {model_name} cluster models",
        )

    for group, grp in iterator:
        if group == "inactive":
            continue

        n_households = grp["ID"].nunique()
        n_rows = len(grp)

        if n_households < min_households or n_rows < min_rows:
            continue

        try:
            split_ds = _split_train_valid(grp)
        except ValueError:
            continue

        train_mask = grp["ds"] < split_ds
        valid_mask = ~train_mask

        train_part = grp.loc[train_mask, ["ID", "ds"]]
        valid_part = grp.loc[valid_mask, ["ID", "ds"]]
        if len(train_part) < min_rows or len(valid_part) == 0:
            continue

        model = get_model(
            model_name,
            use_gpu=use_gpu,
            model_params=model_params,
            xgb_profile=xgb_profile,
        )

        X_train = grp.loc[train_mask, feature_cols]
        y_train = grp.loc[train_mask, "target"].to_numpy(copy=False)
        X_valid = grp.loc[valid_mask, feature_cols]
        y_valid = grp.loc[valid_mask, "target"].to_numpy(copy=False)

        eval_log = _fit_model(
            model=model,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            verbose=verbose,
        )

        metadata = _build_fit_metadata(
            train_part=train_part,
            valid_part=valid_part,
            split_ds=split_ds,
            feature_cols=feature_cols,
            model_name=model_name,
            use_gpu=use_gpu,
            extra={
                "scope": "cluster",
                "forecast_group": str(group),
                "n_households": n_households,
                "n_rows": n_rows,
                "xgb_profile": xgb_profile if model_name.lower() == "xgb" else None,
                "model_params": model_params or {},
            },
        )

        model_dict[group] = {
            "model": model,
            "eval_log": eval_log,
            "metadata": metadata,
        }

    return model_dict
