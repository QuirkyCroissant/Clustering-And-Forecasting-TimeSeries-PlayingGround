def get_model(model_name: str, random_state: int = 42):
    model_name = model_name.lower()

    if model_name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            objective="mae",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        )

    if model_name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=random_state,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def fit_global_model(train_df, feature_cols, model_name="lgbm"):
    model = get_model(model_name)
    X = train_df[feature_cols].fillna(0.0)
    y = train_df["target"].values
    model.fit(X, y)
    return model


def fit_cluster_models(train_df, feature_cols, model_name="lgbm", min_rows=500):
    models = {}
    counts = train_df["ForecastGroup"].value_counts()

    for group, grp in train_df.groupby("ForecastGroup"):
        if group == "inactive":
            continue
        if counts[group] < min_rows:
            continue

        model = get_model(model_name)
        X = grp[feature_cols].fillna(0.0)
        y = grp["target"].values
        model.fit(X, y)
        models[group] = model

    return models