

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fname = "data/AirQualityUCI.csv"

df = pd.read_csv(fname, sep=";", decimal=",")

#drop unnamed empty columns if present
df = df[[c for c in df.columns if not c.startswith("Unnamed")]]

df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H.%M.%S"
)

measurement_cols = [c for c in df.columns
                    if c not in ["Date", "Time", "Datetime"]]

# Replace -200 (dataset missing flag) with NaN
df[measurement_cols] = df[measurement_cols].replace(-200, np.nan)

#Figure 1
plt.figure(figsize=(14, 8))

for col in measurement_cols:
    plt.plot(df["Datetime"], df[col], label=col, alpha=0.6)

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Air Quality Measurements Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
plt.tight_layout()
plt.savefig("output/figure1_measurements_over_time.png", dpi=200)
plt.close()


df["hour"] = df["Datetime"].dt.hour
df["DayNight"] = np.where(df["hour"].between(8, 19), "Day", "Night")

#Figure 2
means_daynight = df.groupby("DayNight")[measurement_cols].mean()
stds_daynight = df.groupby("DayNight")[measurement_cols].std()

fig, ax = plt.subplots(figsize=(14, 6))
idx = np.arange(len(measurement_cols))
width = 0.35

ax.bar(idx - width/2, means_daynight.loc["Day"],
       width, yerr=stds_daynight.loc["Day"],
       label="Day", capsize=3)
ax.bar(idx + width/2, means_daynight.loc["Night"],
       width, yerr=stds_daynight.loc["Night"],
       label="Night", capsize=3)

ax.set_xticks(idx)
ax.set_xticklabels(measurement_cols, rotation=45, ha="right")
ax.set_ylabel("Value")
ax.set_title("Average Values for Day and Night (±1 SD)")
ax.legend()
plt.tight_layout()
plt.savefig("output/figure2_day_night.png", dpi=200)
plt.close()

df["WeekdayNum"] = df["Datetime"].dt.weekday
df["DayType"] = np.where(df["WeekdayNum"] >= 5, "Weekend", "Weekday")

#Figure 3
means_daytype = df.groupby("DayType")[measurement_cols].mean()
stds_daytype = df.groupby("DayType")[measurement_cols].std()

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(idx - width/2, means_daytype.loc["Weekday"],
       width, yerr=stds_daytype.loc["Weekday"],
       label="Weekday", capsize=3)
ax.bar(idx + width/2, means_daytype.loc["Weekend"],
       width, yerr=stds_daytype.loc["Weekend"],
       label="Weekend", capsize=3)

ax.set_xticks(idx)
ax.set_xticklabels(measurement_cols, rotation=45, ha="right")
ax.set_ylabel("Value")
ax.set_title("Average Values for Weekdays and Weekends (±1 SD)")
ax.legend()
plt.tight_layout()
plt.savefig("output/figure3_weekday_weekend.png", dpi=200)
plt.close()

#Figure 4
stds = df[measurement_cols].std().sort_values()

plt.figure(figsize=(12, 6))
stds.plot(kind="bar")
plt.ylabel("Standard Deviation")
plt.title("Volatility of Pollution and Weather Measures")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("output/figure4_volatility.png", dpi=200)
plt.close()

#Figure 5
corr = df[measurement_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Pollution and Weather Measures")
plt.tight_layout()
plt.savefig("output/figure5_correlation_heatmap.png", dpi=200)
plt.close()
