from tqdm.auto import tqdm
from lightgbm import log_evaluation, early_stopping
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

def fit_global_model(train_df, feature_cols, model_name="lgbm", use_gpu=False):
    model = get_model(model_name, use_gpu=use_gpu)

    train_df = train_df.sort_values(["ds", "ID"]).reset_index(drop=True)
    unique_ds = np.sort(train_df["ds"].unique())
    split_idx = int(len(unique_ds) * 0.8)
    split_ds = unique_ds[split_idx]

    train_part = train_df[train_df["ds"] < split_ds]
    valid_part = train_df[train_df["ds"] >= split_ds]

    X_train = train_part[feature_cols].fillna(0.0)
    y_train = train_part["target"].values
    X_valid = valid_part[feature_cols].fillna(0.0)
    y_valid = valid_part["target"].values

    if model_name == "xgb":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=50,
        )
        eval_log = model.evals_result()
    else:
        model.fit(X_train, y_train)
        eval_log = {}

    return {
        "model": model,
        "eval_log": eval_log,
        "n_rows_train": len(train_part),
        "n_rows_valid": len(valid_part),
        "n_households": train_df["ID"].nunique(),
    }

def fit_cluster_models(
    train_df,
    feature_cols,
    model_name="lgbm",
    min_households=30,
    min_rows=500,
    use_gpu=False,
    show_progress=True,
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

        grp = grp.sort_values(["ds", "ID"]).reset_index(drop=True)
        unique_ds = np.sort(grp["ds"].unique())
        if len(unique_ds) < 10:
            continue

        split_idx = int(len(unique_ds) * 0.8)
        split_ds = unique_ds[split_idx]

        train_part = grp[grp["ds"] < split_ds]
        valid_part = grp[grp["ds"] >= split_ds]

        if len(train_part) < min_rows or len(valid_part) == 0:
            continue

        model = get_model(model_name, use_gpu=use_gpu)

        X_train = train_part[feature_cols].fillna(0.0)
        y_train = train_part["target"].values
        X_valid = valid_part[feature_cols].fillna(0.0)
        y_valid = valid_part["target"].values

        if model_name == "xgb":
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )
            eval_log = model.evals_result()
        else:
            model.fit(X_train, y_train)
            eval_log = {}

        model_dict[group] = {
            "model": model,
            "eval_log": eval_log,
            "n_households": n_households,
            "n_rows_train": len(train_part),
            "n_rows_valid": len(valid_part),
        }

    return model_dict