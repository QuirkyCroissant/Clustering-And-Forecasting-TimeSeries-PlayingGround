from pathlib import Path
import pandas as pd


def load_wide_csv(path: Path) -> pd.DataFrame:
    """Load a wide household-by-day table and normalise the key column to ID."""

    df = pd.read_csv(path)
    first_col = df.columns[0]
    if first_col != "ID":
        df = df.rename(columns={first_col: "ID"})
    return df


def load_cluster_labels(path: Path) -> pd.DataFrame:
    """Load final cluster labels and derive the ForecastGroup used by forecasting."""

    df = pd.read_csv(path)
    required = {"ID", "RefinedCluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cluster file missing columns: {missing}")

    out = df.copy()
    out["ForecastGroup"] = out["RefinedCluster"].astype(str)
    out.loc[out["RefinedCluster"] == -1, "ForecastGroup"] = "inactive"
    return out


def load_static_features(path: Path) -> pd.DataFrame:
    """Load optional static features and enforce the expected household key."""

    df = pd.read_csv(path)
    if "ID" not in df.columns:
        raise ValueError("Static feature file must contain an ID column")
    return df


def discover_cluster_cases(cluster_dir: Path, pattern: str = "case*_clusters.csv") -> dict[str, Path]:
    """Discover saved cluster case files and return a casename to path lookup."""

    case_files = sorted(cluster_dir.glob(pattern))
    if not case_files:
        raise FileNotFoundError(f"No cluster case files matching {pattern!r} found in {cluster_dir}")

    return {
        path.stem.replace("_clusters", ""): path
        for path in case_files
    }


def ensure_output_dirs(repo_root: Path) -> dict:
    """Create the shared output folders used by notebook-level forecasting runs."""
    
    paths = {
        "metrics": repo_root / "outputs" / "metrics",
        "plots": repo_root / "outputs" / "plots",
        "predictions": repo_root / "outputs" / "predictions",
        "models": repo_root / "outputs" / "models",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_experiment_dirs(repo_root: Path, experiment_name: str) -> dict:
    """Create an isolated output tree for one named experiment run."""

    root = repo_root / "outputs" / "experiments" / experiment_name
    paths = {
        "root": root,
        "metrics": root / "metrics",
        "plots": root / "plots",
        "predictions": root / "predictions",
        "metadata": root / "metadata",
        "models": root / "models",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
