"""Correlation analysis for day/night and weekday/weekend flags.

This script loads the labeled dataset and computes Pearson correlations
between the Day/Weekday flags and each pollutant.

Outputs a CSV and a heatmap for quick visualization.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

#Compute correlations and p-values for each pollutant vs day and weekday
def compute_day_week_correlations(df: pd.DataFrame, measurement_cols: list[str]) -> pd.DataFrame:
    results = []
    for meas in measurement_cols:

        # drop missing values and align with flags
        df_sub = df[["Day", "Weekday", meas]].dropna()

        if df_sub.shape[0] < 50:
            continue

        # compute Pearson correlation and p-values
        r_day, p_day = pearsonr(df_sub["Day"], df_sub[meas])
        r_weekday, p_weekday = pearsonr(df_sub["Weekday"], df_sub[meas])

        results.append({
            "pollutant": meas,
            "r_day": r_day,
            "p_day": p_day,
            "r_weekday": r_weekday,
            "p_weekday": p_weekday,
            "n": df_sub.shape[0],
        })

    out = pd.DataFrame(results)

    # sort by strongest correlation with day (absolute value)
    out = out.sort_values(by="r_day", key=lambda s: np.abs(s), ascending=False)

    return out

#Mark stronger and statistically significant relationships
def highlight_strong_correlations(corr_df: pd.DataFrame) -> pd.DataFrame:
    corr_df = corr_df.copy()

    # flag stronger correlations (using 0.3 cutoff)
    corr_df["strong_day"] = (np.abs(corr_df["r_day"]) >= 0.3) & (corr_df["p_day"] < 0.05)
    corr_df["strong_weekday"] = (np.abs(corr_df["r_weekday"]) >= 0.3) & (corr_df["p_weekday"] < 0.05)

    return corr_df

#Create a simple heatmap of correlations for visualization
def plot_correlation_heatmap(corr_df: pd.DataFrame, output_path: str) -> None:
    heatmap_df = corr_df.set_index("pollutant")[["r_day", "r_weekday"]]

    plt.figure(figsize=(8, max(4, len(heatmap_df) * 0.3)))

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlations with Day and Weekday")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    # set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "output")

    os.makedirs(output_dir, exist_ok=True)

    # load labeled dataset
    input_path = os.path.join(data_dir, "day_weekday_data_flags.csv")
    print(f"Reading data from: {input_path}")
    df = pd.read_csv(input_path)

    # select measurement columns (exclude metadata + flags)
    measurement_cols = [
        c for c in df.columns
        if c not in ["Date", "Time", "Datetime", "Day", "Weekday"]
    ]

    # compute correlations
    corr_df = compute_day_week_correlations(df, measurement_cols)
    corr_df = highlight_strong_correlations(corr_df)

    # save results
    out_csv = os.path.join(output_dir, "day_weekday_correlations.csv")
    corr_df.to_csv(out_csv, index=False)
    print(f"Saved results to: {out_csv}")

    # print quick summary
    print("\nStronger correlations (|r| >= 0.3 and p < 0.05):")
    strong = corr_df[(corr_df["strong_day"] | corr_df["strong_weekday"])]

    if strong.empty:
        print("None found")
    else:
        print(strong[["pollutant", "r_day", "p_day", "r_weekday", "p_weekday"]].to_string(index=False))

    # save heatmap
    out_png = os.path.join(output_dir, "day_weekday_correlations.png")
    plot_correlation_heatmap(corr_df, out_png)
    print(f"Saved heatmap to: {out_png}")


if __name__ == "__main__":
    main()