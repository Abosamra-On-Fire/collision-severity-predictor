import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config as cfg


def get_pie_chart_for_collision_severity(df: pd.DataFrame, save_path: str):
    counts = df["collision_severity"].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=["Slight", "Serious", "Fatal"], autopct="%1.1f%%", startangle=20)

    plt.title("Accident Distribution by Collision Severity")
    plt.axis("equal")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def geographic_distribution_accidents(df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(8, 10))
    plt.scatter(df["longitude"], df["latitude"], alpha=0.05, s=0.5, c="navy")
    plt.title("Geographic Distribution of Accidents")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(save_path, dpi=150)


def accidents_count_and_severity_per_hour(df: pd.DataFrame, save_path: str):
    df["hour"] = pd.to_datetime(df["time"].astype(str), format="%H:%M:%S", errors="coerce").dt.hour
    hourly = df.groupby("hour")["collision_severity"].agg(["mean", "count"])
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(hourly.index, hourly["count"], alpha=0.4, label="Count")
    ax2 = ax1.twinx()
    ax2.plot(hourly.index, hourly["mean"], color="red", marker="o", label="Avg Severity")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Accident Count")
    ax2.set_ylabel("Avg Severity (higher = worse)")
    plt.title("Accident Count & Severity by Hour")
    plt.savefig(save_path)


def rain_vs_collision_severity(df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(8, 6))

    sns.boxplot(x="collision_severity", y="wx_rain", data=df[df["wx_rain"] > 0])

    plt.title("Rain vs Collision Severity")
    plt.xlabel("Collision Severity")
    plt.ylabel("Rain")
    plt.ylim(df["wx_rain"].quantile(0.01), df["wx_rain"].quantile(0.99))
    plt.savefig(save_path)


def avg_collision_severity_by_road_type(df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(10, 6))

    sns.barplot(x="road_type", y="collision_severity", data=df, estimator="mean")

    plt.xticks(
        ticks=range(len(df["road_type"].unique())),
        labels=[
            "Roundabout",
            "One Way Street",
            "Dual Carriage Way",
            "Single Carriage Way",
            "Slip Road",
            "Unknown",
        ],
        rotation=15,
    )
    plt.title("Average Collision Severity by Road Type")
    plt.xlabel("Road Type")
    plt.ylabel("Avg Severity")
    plt.savefig(save_path)


def main_visualize():
    df = pd.read_csv(rf"{cfg.INTERIM_DATA_DIR}\{cfg.CLEANED_TRAIN_OUTPUT_FILE}")
    get_pie_chart_for_collision_severity(df, rf"{cfg.FIGURES_DIR}\{cfg.PIE_CHART_FOR_SEVERITY}")
    geographic_distribution_accidents(df, rf"{cfg.FIGURES_DIR}\{cfg.GEO_DISTRIBUTION}")
    accidents_count_and_severity_per_hour(df, rf"{cfg.FIGURES_DIR}\{cfg.HOURLY_ACCIDENTS}")
    rain_vs_collision_severity(df, rf"{cfg.FIGURES_DIR}\{cfg.RAIN_VS_SEVERITY}")
    avg_collision_severity_by_road_type(df, rf"{cfg.FIGURES_DIR}\{cfg.AVG_COLLISION_VS_ROAD_TYPE}")


if __name__ == "__main__":
    main_visualize()
