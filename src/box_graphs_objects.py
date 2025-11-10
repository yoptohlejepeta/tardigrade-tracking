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
        "video_ids": [1, 2, 3, 4, 5, 31, 33, 10, 11],
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

boxplot_data = []
labels = []
colors = []

for config in configs:
    group = config["group"]
    abb = config["abb"]
    subgroup = config["subgroup"]
    video_ids = config["video_ids"]
    color = config["color"]

    loginfo(f"Processing {group} - {subgroup}...")

    object_means = []

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

        track_means = valid_data.groupby("track_id")["displacement"].mean()
        object_means.extend(track_means.tolist())

    loginfo(f"  Total objects: {len(object_means)}")
    boxplot_data.append(object_means)
    labels.append(f"{group}\n{subgroup}")
    colors.append(color)

fig, ax = plt.subplots(figsize=(10, 6))

bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True, widths=0.6)

for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

for i, (data, color) in enumerate(zip(boxplot_data, colors)):
    x = np.random.normal(i + 1, 0.04, size=len(data))
    ax.scatter(
        x,
        data,
        alpha=0.6,
        s=20,
        color=color,
        edgecolors="black",
        linewidth=0.5,
        zorder=3,
    )

ax.set_ylabel("Mean Displacement", fontsize=12)
ax.set_title("Mean Displacement per Object by Group", fontsize=14)
ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("displacement_boxplot_objects.png", dpi=150)
plt.close()

loginfo("\nSaved: displacement_boxplot_objects.png")
loginfo("Done!")
