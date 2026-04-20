import pandas as pd


def mae_by_household(pred_wide: pd.DataFrame, truth_wide: pd.DataFrame) -> pd.DataFrame:
    pred = pred_wide.copy()
    truth = truth_wide.copy()

    pred = pred.rename(columns={pred.columns[0]: "ID"})
    truth = truth.rename(columns={truth.columns[0]: "ID"})

    pred = pred.set_index("ID").sort_index()
    truth = truth.set_index("ID").sort_index()

    common_cols = [c for c in truth.columns if c in pred.columns]
    mae = (truth[common_cols] - pred[common_cols]).abs().mean(axis=1)

    out = mae.rename("MAE").reset_index()
    return out


def summarise_mae(mae_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_households": [len(mae_df)],
            "mean_mae": [mae_df["MAE"].mean()],
            "median_mae": [mae_df["MAE"].median()],
            "std_mae": [mae_df["MAE"].std()],
        }
    )