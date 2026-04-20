from pathlib import Path
import pandas as pd


def load_wide_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    first_col = df.columns[0]
    if first_col != "ID":
        df = df.rename(columns={first_col: "ID"})
    return df


def load_cluster_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ID", "RefinedCluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cluster file missing columns: {missing}")

    out = df[["ID", "RefinedCluster"]].copy()
    out["ForecastGroup"] = out["RefinedCluster"].astype(str)
    out.loc[out["RefinedCluster"] == -1, "ForecastGroup"] = "inactive"
    return out


def load_static_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ID" not in df.columns:
        raise ValueError("Static feature file must contain an ID column")
    return df


def ensure_output_dirs(repo_root: Path) -> dict:
    paths = {
        "metrics": repo_root / "outputs" / "metrics",
        "plots": repo_root / "outputs" / "plots",
        "predictions": repo_root / "outputs" / "predictions",
        "models": repo_root / "outputs" / "models",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths