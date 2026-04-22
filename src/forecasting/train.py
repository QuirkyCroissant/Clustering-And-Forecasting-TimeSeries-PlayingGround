from tqdm.auto import tqdm
import numpy as np


def get_model(model_name: str, random_state: int = 42, use_gpu: bool = False):
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
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=random_state,
            verbosity=1,
        )
        if use_gpu:
            params["device"] = "cuda"
        return XGBRegressor(**params)

    raise ValueError(f"Unsupported model_name: {model_name}")


def _split_train_valid(df):
    df = df.sort_values(["ds", "ID"]).reset_index(drop=True)
    unique_ds = np.sort(df["ds"].unique())

    if len(unique_ds) < 2:
        raise ValueError("Need at least two unique ds values to create a validation split")

    split_idx = int(len(unique_ds) * 0.8)
    split_idx = min(max(split_idx, 1), len(unique_ds) - 1)
    split_ds = unique_ds[split_idx]

    train_part = df[df["ds"] < split_ds]
    valid_part = df[df["ds"] >= split_ds]
    return train_part, valid_part, split_ds


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
    verbose=50,
):
    model = get_model(model_name, use_gpu=use_gpu)

    train_part, valid_part, split_ds = _split_train_valid(train_df)

    X_train = train_part[feature_cols].fillna(0.0)
    y_train = train_part["target"].values
    X_valid = valid_part[feature_cols].fillna(0.0)
    y_valid = valid_part["target"].values

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
        extra={"scope": "global"},
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
    show_progress=True,
    verbose=False,
):
    model_dict = {}
    groups = list(train_df.groupby("ForecastGroup"))

    iterator = groups
    if show_progress:
        iterator = tqdm(groups, total=len(groups), desc=f"Training {model_name} cluster models")

    for group, grp in iterator:
        if group == "inactive":
            continue

        n_households = grp["ID"].nunique()
        n_rows = len(grp)

        if n_households < min_households or n_rows < min_rows:
            continue

        try:
            train_part, valid_part, split_ds = _split_train_valid(grp)
        except ValueError:
            continue

        if len(train_part) < min_rows or len(valid_part) == 0:
            continue

        model = get_model(model_name, use_gpu=use_gpu)

        X_train = train_part[feature_cols].fillna(0.0)
        y_train = train_part["target"].values
        X_valid = valid_part[feature_cols].fillna(0.0)
        y_valid = valid_part["target"].values

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
            },
        )

        model_dict[group] = {
            "model": model,
            "eval_log": eval_log,
            "metadata": metadata,
        }

    return model_dict
