import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.info import loginfo

configs = [
    {
        "group": "Control",
        "subgroup": "0.5",
        "abb": "C",
        "video_ids": [7, 11, 13, 22, 8],
        "color": "#1f77b4",
    },
    {
        "group": "Control",
        "subgroup": "1.0",
        "abb": "C",
        "video_ids": [4, 6, 25, 28, 33],
        "color": "#ff7f0e",
    },
    {
        "group": "CytochalasinD",
        "subgroup": "20",
        "abb": "CD",
        "video_ids": [1, 10, 11, 14, 19],
        "color": "#d62728",
    },
    {
        "group": "CytochalasinD",
        "subgroup": "50",
        "abb": "CD",
        "video_ids": [9, 14, 19, 25, 26],
        "color": "#9467bd",
    },
    {
        "group": "Taxol",
        "subgroup": "200",
        "abb": "T",
        "video_ids": [1, 2, 3, 4, 5],
        "color": "#5318b9",
    },
    {
        "group": "Taxol",
        "subgroup": "100",
        "abb": "T",
        "video_ids": [14, 16, 25, 28, 7],
        "color": "#c10eaf",
    },
    {
        "group": "Nocodazole",
        "subgroup": "10",
        "abb": "N",
        "video_ids": [10, 11, 7, 8, 9],
        "color": "#7f7f7f",
    },
    {
        "group": "Nocodazole",
        "subgroup": "1",
        "abb": "N",
        "video_ids": [32, 33, 34, 35, 30, 39, 41, 23],
        "color": "#dde624",
    },
]

all_displacement_stats = []
all_area_stats = []

for config in configs:
    group = config["group"]
    abb = config["abb"]
    subgroup = config["subgroup"]
    video_ids = config["video_ids"]
    color = config["color"]

    loginfo(f"Processing {group} - {subgroup}...")

    dfs = []

    for video_id in video_ids:
        video = f"{abb}{video_id}.tuns"
        df_video = pd.read_csv(f"./csvs/{group}/{video}/{video}.csv")
        track_ids_in_all_frames = df_video["track_id"].value_counts()
        num_frames = df_video["frame"].nunique()

        track_ids_in_all_frames = track_ids_in_all_frames[
            track_ids_in_all_frames == num_frames
        ].index.tolist()
        valid_data = df_video[df_video["track_id"].isin(track_ids_in_all_frames)]
        valid_data = valid_data.sort_values(by=["track_id", "frame"])
        valid_data["dx"] = valid_data.groupby("track_id")["centroid_x"].diff().fillna(0)
        valid_data["dy"] = valid_data.groupby("track_id")["centroid_y"].diff().fillna(0)
        valid_data["displacement"] = np.sqrt(
            valid_data["dx"] ** 2 + valid_data["dy"] ** 2
        )
        valid_data["video"] = video

        dfs.append(valid_data)

    df = pd.concat(dfs, ignore_index=True)

    displacement_stats = df.groupby("frame")["displacement"].agg(
        ["mean", "std", "count"]
    )
    displacement_stats["se"] = displacement_stats["std"] / np.sqrt(
        displacement_stats["count"]
    )
    displacement_stats["group"] = group
    displacement_stats["subgroup"] = subgroup
    displacement_stats["color"] = color
    displacement_stats["label"] = f"{group} - {subgroup}"
    all_displacement_stats.append(displacement_stats)

    area_stats = df.groupby("frame")["area"].agg(["mean", "std", "count"])
    area_stats["se"] = area_stats["std"] / np.sqrt(area_stats["count"])
    area_stats["group"] = group
    area_stats["subgroup"] = subgroup
    area_stats["color"] = color
    area_stats["label"] = f"{group} - {subgroup}"
    all_area_stats.append(area_stats)

# ============================================================================
# DISPLACEMENT CHART
# ============================================================================
plt.figure(figsize=(16, 6))

for stats in all_displacement_stats:
    plt.plot(
        stats.index,
        stats["mean"],
        linewidth=2,
        label=stats["label"].iloc[0],
        color=stats["color"].iloc[0],
    )
    plt.fill_between(
        stats.index,
        stats["mean"] - stats["se"],
        stats["mean"] + stats["se"],
        alpha=0.2,
        color=stats["color"].iloc[0],
    )

plt.xlabel("Frame", fontsize=12)
plt.ylabel("Displacement", fontsize=12)
plt.title("Displacement over Frames - All Groups", fontsize=14)
plt.xticks(range(0, max([s.index.max() for s in all_displacement_stats]) + 1, 1000))
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("displacement_all_groups.png", dpi=150)
plt.close()

loginfo("Saved: displacement_all_groups.png")

# ============================================================================
# AREA CHART
# ============================================================================
plt.figure(figsize=(16, 6))

for stats in all_area_stats:
    plt.plot(
        stats.index,
        stats["mean"],
        linewidth=2,
        label=stats["label"].iloc[0],
        color=stats["color"].iloc[0],
    )
    plt.fill_between(
        stats.index,
        stats["mean"] - stats["se"],
        stats["mean"] + stats["se"],
        alpha=0.2,
        color=stats["color"].iloc[0],
    )

plt.xlabel("Frame", fontsize=12)
plt.ylabel("Area", fontsize=12)
plt.title("Area over Frames - All Groups", fontsize=14)
plt.xticks(range(0, max([s.index.max() for s in all_area_stats]) + 1, 1000))
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("area_all_groups.png", dpi=150)
plt.close()

loginfo("Saved: area_all_groups.png")
loginfo("Done!")
