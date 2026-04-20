from tqdm.auto import tqdm
from lightgbm import log_evaluation, early_stopping

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

def fit_global_model(train_df, feature_cols, model_name="lgbm", use_gpu=False):
    model = get_model(model_name, use_gpu=use_gpu)

    split_idx = int(len(train_df) * 0.8)
    train_part = train_df.iloc[:split_idx]
    valid_part = train_df.iloc[split_idx:]

    X_train = train_part[feature_cols].fillna(0.0)
    y_train = train_part["target"].values
    X_valid = valid_part[feature_cols].fillna(0.0)
    y_valid = valid_part["target"].values

    if model_name == "lgbm":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="l1",
            callbacks=[
                log_evaluation(period=50),
                early_stopping(stopping_rounds=100),
            ],
        )
    elif model_name == "xgb":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=50,
        )
    else:
        model.fit(X_train, y_train)

    return model

def fit_cluster_models(
    train_df,
    feature_cols,
    model_name="lgbm",
    min_rows=500,
    use_gpu=False,
    show_progress=True,
):
    models = {}
    counts = train_df["ForecastGroup"].value_counts()
    groups = list(train_df.groupby("ForecastGroup"))

    iterator = groups
    if show_progress:
        iterator = tqdm(groups, total=len(groups), desc=f"Training {model_name} cluster models")

    for group, grp in iterator:
        if group == "inactive":
            continue
        if counts[group] < min_rows:
            print(f"Skipping {group} because it has only {counts[group]} rows")
            continue

        print(f"Training group {group} with {len(grp):,} rows")
        model = get_model(model_name, use_gpu=use_gpu)
        X = grp[feature_cols].fillna(0.0)
        y = grp["target"].values
        model.fit(X, y)
        models[group] = model

    print(f"Finished {len(models)} cluster models")
    return models