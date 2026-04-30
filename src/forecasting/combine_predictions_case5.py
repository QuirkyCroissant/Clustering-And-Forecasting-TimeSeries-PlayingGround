#!/usr/bin/env python3
"""
Combine AGP and XGB predictions per cluster size.

Usage:
  python scripts/combine_predictions_case5.py \
    --agp outputs/agp_forecast_case5_clusters_cluster_specific_combined.csv \
    --xgb outputs/pred_2024_cluster_xgb(13).csv \
    --clusters outputs/feature/case5_clusters.csv \
    --out outputs/pred_combined_case5.csv

The script: 
- computes cluster sizes from the clusters file
- identifies "big" clusters (> --big-threshold, default 500)
- for IDs in big clusters, uses XGB prediction when available
- otherwise, uses AGP prediction

The script is chunked and memory-conscious.
"""
from __future__ import annotations

import argparse
from typing import Optional, Tuple

import pandas as pd


def detect_column(cols: pd.Index, candidates: Tuple[str, ...]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agp", required=True)
    p.add_argument("--xgb", required=True)
    p.add_argument("--clusters", required=True)
    p.add_argument("--out", default="outputs/pred_combined_case5.csv")
    p.add_argument("--chunksize", type=int, default=200_000)
    p.add_argument("--big-threshold", type=int, default=500)
    args = p.parse_args()

    agp_path = args.agp
    xgb_path = args.xgb
    clusters_path = args.clusters
    out_path = args.out
    chunksize = args.chunksize
    big_threshold = args.big_threshold

    print(f"Clusters: {clusters_path}\nAGP: {agp_path}\nXGB: {xgb_path}\nOut: {out_path}")

    # Read clusters
    df_clusters = pd.read_csv(clusters_path)
    # normalize cluster id / ID columns
    id_col = detect_column(df_clusters.columns, ("ID", "Id", "id", "series_id", "unique_id"))
    cluster_col = detect_column(df_clusters.columns, ("RefinedCluster", "cluster", "Refined", "cluster_id"))
    if id_col is None or cluster_col is None:
        raise SystemExit("Cannot detect ID or cluster columns in clusters file")

    df_clusters = df_clusters[[id_col, cluster_col]].rename(columns={id_col: "ID", cluster_col: "cluster"})
    df_clusters["ID"] = df_clusters["ID"].astype(str)

    cluster_counts = df_clusters.groupby("cluster")["ID"].nunique().to_dict()
    big_clusters = {c for c, cnt in cluster_counts.items() if cnt > big_threshold}
    is_big_ids = set(df_clusters[df_clusters["cluster"].isin(big_clusters)]["ID"].tolist())

    print(f"Found {len(cluster_counts)} clusters; {len(big_clusters)} are big (> {big_threshold}).")

    # Inspect headers for AGP and XGB to detect id/date/prediction columns
    agp_cols = pd.read_csv(agp_path, nrows=0).columns
    xgb_cols = pd.read_csv(xgb_path, nrows=0).columns

    id_candidates = ("ID", "Id", "id", "series_id", "unique_id")
    date_candidates = ("date", "ds", "timestamp", "day")
    pred_candidates = ("y_pred", "yhat", "forecast", "pred", "prediction", "mean", "pred_mean")

    agp_id = detect_column(agp_cols, id_candidates)
    xgb_id = detect_column(xgb_cols, id_candidates)
    # prefer date col if present
    agp_date = detect_column(agp_cols, date_candidates)
    xgb_date = detect_column(xgb_cols, date_candidates)

    agp_pred = detect_column(agp_cols, pred_candidates)
    xgb_pred = detect_column(xgb_cols, pred_candidates)

    if agp_id is None or xgb_id is None:
        raise SystemExit("Could not detect ID column in one of the prediction files")

    # If prediction column not found, try to infer by excluding known columns
    def infer_pred(cols, idc, datec):
        cols_list = [c for c in cols if c not in {idc, datec, "cluster"}]
        # choose a numeric-looking candidate
        for cand in pred_candidates:
            if cand in cols_list:
                return cand
        # fallback to the last column
        return cols_list[-1] if cols_list else None

    if agp_pred is None:
        agp_pred = infer_pred(list(agp_cols), agp_id, agp_date)
    if xgb_pred is None:
        xgb_pred = infer_pred(list(xgb_cols), xgb_id, xgb_date)

    print(f"AGP id/date/pred: {agp_id}/{agp_date}/{agp_pred}")
    print(f"XGB id/date/pred: {xgb_id}/{xgb_date}/{xgb_pred}")

    # Build XGB map only for big IDs
    xgb_map = {}
    read_cols = [xgb_id, xgb_pred] + ([xgb_date] if xgb_date else [])
    for chunk in pd.read_csv(xgb_path, usecols=read_cols, chunksize=chunksize):
        chunk[xgb_id] = chunk[xgb_id].astype(str)
        if xgb_date:
            keys = chunk[xgb_id] + "|" + chunk[xgb_date].astype(str)
        else:
            keys = chunk[xgb_id].astype(str)
        mask = chunk[xgb_id].isin(is_big_ids)
        sub = chunk.loc[mask]
        if xgb_date:
            sub_keys = sub[xgb_id].astype(str) + "|" + sub[xgb_date].astype(str)
        else:
            sub_keys = sub[xgb_id].astype(str)
        xgb_map.update(dict(zip(sub_keys.tolist(), sub[xgb_pred].tolist())))

    print(f"Loaded XGB map entries for big-cluster IDs: {len(xgb_map)}")

    # Process AGP file in chunks and write final predictions
    first_write = True
    total_rows = 0
    replaced_with_xgb = 0

    for chunk in pd.read_csv(agp_path, chunksize=chunksize):
        # normalize ID and date
        chunk[agp_id] = chunk[agp_id].astype(str)
        if agp_date:
            keys = chunk[agp_id] + "|" + chunk[agp_date].astype(str)
        else:
            keys = chunk[agp_id].astype(str)

        use_xgb_mask = chunk[agp_id].isin(is_big_ids)
        # default to AGP prediction value
        chunk_cols = list(chunk.columns)
        # ensure prediction column exists
        if agp_pred not in chunk_cols:
            raise SystemExit(f"AGP prediction column not detected: {agp_pred}")
        # create final_pred column defaulting to AGP
        chunk["final_pred"] = chunk[agp_pred]

        # For rows in big clusters, if XGB has a value for exact key, use it
        if use_xgb_mask.any():
            selected_idx = chunk.index[use_xgb_mask]
            for idx in selected_idx:
                k = keys.iloc[idx]
                if k in xgb_map:
                    chunk.at[idx, "final_pred"] = xgb_map[k]
                    replaced_with_xgb += 1

        # write chunk
        mode = "w" if first_write else "a"
        header = first_write
        chunk.to_csv(out_path, index=False, mode=mode, header=header)
        first_write = False
        total_rows += len(chunk)

    print(f"Wrote {total_rows} rows to {out_path}; replaced {replaced_with_xgb} rows with XGB predictions.")


if __name__ == "__main__":
    main()
