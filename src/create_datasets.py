import pandas as pd
import numpy as np
from src.info import loginfo

configs = [
    {
        "group": "Control",
        "subgroup": "0.5",
        "abb": "C",
        "video_ids": [7, 11, 13, 22, 8],
    },
    {
        "group": "Control",
        "subgroup": "1.0",
        "abb": "C",
        "video_ids": [4, 6, 25, 28, 33],
    },
    {
        "group": "CytochalasinD",
        "subgroup": "20",
        "abb": "CD",
        "video_ids": [1, 2, 3, 4, 5, 31, 33, 10, 11],
    },
    {
        "group": "CytochalasinD",
        "subgroup": "50",
        "abb": "CD",
        "video_ids": [9, 14, 19, 25, 26],
    },
    {
        "group": "Taxol",
        "subgroup": "200",
        "abb": "T",
        "video_ids": [14, 16, 25, 28, 30, 7],
    },
    {
        "group": "Taxol",
        "subgroup": "100",
        "abb": "T",
        "video_ids": [1, 2, 3, 4, 5, 19, 32],
    },
    {
        "group": "Nocodazole",
        "subgroup": "10",
        "abb": "N",
        "video_ids": [7, 8, 9, 10, 11, 23, 30],
    },
    {
        "group": "Nocodazole",
        "subgroup": "1",
        "abb": "N",
        "video_ids": [32, 33, 34, 35, 39, 41],
    },
]

# Define objects to exclude
EXCLUSIONS = {
    "C4.tuns": [4],
    "C11.tuns": [10, 32, 33],
    "C13.tuns": [7, 10, 11],
    "C25.tuns": [20, 27],
    "C33.tuns": [18, 10, 6],
    "CD2.tuns": [17],
    "CD3.tuns": [2],
    "CD4.tuns": [4, 13, 18],
    "CD9.tuns": [1],
    "CD19.tuns": [18, 7, 9, 16],
    "CD26.tuns": [3],
    "T25.tuns": [28],
    "N8.tuns": [1],
    "N30.tuns": [30],
    "N32.tuns": [13],
    "N33.tuns": [10, 12],
    "N34.tuns": [13, 22, 14, 17, 19],
    "N35.tuns": [2, 3],
    "N39.tuns": [23],
}

PIXEL_TO_UM = 3.33
PIXEL2_TO_UM2 = PIXEL_TO_UM**2

boxplot_records = []
timeline_records = []

for config in configs:
    group = config["group"]
    abb = config["abb"]
    subgroup = config["subgroup"]
    video_ids = config["video_ids"]

    loginfo(f"Processing {group} - {subgroup}...")

    dfs = []

    for video_id in video_ids:
        video = f"{abb}{video_id}.tuns"
        df_video = pd.read_csv(f"./csvs/{group}/{video}/{video}.csv")

        # Apply exclusions for this video
        if video in EXCLUSIONS:
            excluded_track_ids = EXCLUSIONS[video]
            df_video = df_video[~df_video["track_id"].isin(excluded_track_ids)]
            loginfo(f"  Excluded track_ids {excluded_track_ids} from {video}")

        track_ids_in_all_frames = df_video["track_id"].value_counts()
        num_frames = df_video["frame"].nunique()

        track_ids_in_all_frames = track_ids_in_all_frames[
            track_ids_in_all_frames == num_frames
        ].index.tolist()
        valid_data = df_video[df_video["track_id"].isin(track_ids_in_all_frames)]
        valid_data = valid_data.sort_values(by=["track_id", "frame"])

        # Calculate displacement
        valid_data["dx"] = valid_data.groupby("track_id")["centroid_x"].diff().fillna(0)
        valid_data["dy"] = valid_data.groupby("track_id")["centroid_y"].diff().fillna(0)
        valid_data["displacement"] = np.sqrt(
            valid_data["dx"] ** 2 + valid_data["dy"] ** 2
        )
        valid_data["video"] = video

        # Collect per-object means for boxplot data
        for track_id in track_ids_in_all_frames:
            track_data = valid_data[valid_data["track_id"] == track_id]
            mean_disp = track_data["displacement"].mean()
            mean_area = track_data["area"].mean()
            mean_feret = track_data["max_feret_diameter"].mean()

            boxplot_records.append(
                {
                    "group": group,
                    "subgroup": subgroup,
                    "video": video,
                    "track_id": track_id,
                    "mean_displacement": mean_disp,
                    "mean_displacement_um": mean_disp * PIXEL_TO_UM,
                    "mean_area": mean_area,
                    "mean_area_um2": mean_area * PIXEL2_TO_UM2,
                    "mean_max_feret_diameter": mean_feret,
                    "mean_max_feret_diameter_um": mean_feret * PIXEL_TO_UM,
                }
            )

        loginfo(f"  Valid data points: {len(valid_data)}")
        dfs.append(valid_data)

    # Concatenate all videos for this group/subgroup
    df = pd.concat(dfs, ignore_index=True)

    # Find frames that exist in ALL videos for this group/subgroup
    frames_per_video = df.groupby("video")["frame"].apply(set)
    common_frames = set.intersection(*frames_per_video)
    common_frames = sorted(list(common_frames))
    
    loginfo(f"  Common frames across all videos: {len(common_frames)}")
    
    # Filter to only use common frames
    df = df[df["frame"].isin(common_frames)]

    # Calculate timeline statistics for each frame
    for frame in common_frames:
        frame_data = df[df["frame"] == frame]

        # Displacement stats
        mean_disp = frame_data["displacement"].mean()
        std_disp = frame_data["displacement"].std()
        se_disp = std_disp / np.sqrt(len(frame_data))

        # Area stats
        mean_area = frame_data["area"].mean()
        std_area = frame_data["area"].std()
        se_area = std_area / np.sqrt(len(frame_data))

        # Max feret diameter stats
        mean_feret = frame_data["max_feret_diameter"].mean()
        std_feret = frame_data["max_feret_diameter"].std()
        se_feret = std_feret / np.sqrt(len(frame_data))

        timeline_records.append(
            {
                "group": group,
                "subgroup": subgroup,
                "frame": frame,
                "mean_displacement": mean_disp,
                "mean_displacement_um": mean_disp * PIXEL_TO_UM,
                "std_displacement": std_disp,
                "std_displacement_um": std_disp * PIXEL_TO_UM,
                "se_displacement": se_disp,
                "se_displacement_um": se_disp * PIXEL_TO_UM,
                "mean_area": mean_area,
                "mean_area_um2": mean_area * PIXEL2_TO_UM2,
                "std_area": std_area,
                "std_area_um2": std_area * PIXEL2_TO_UM2,
                "se_area": se_area,
                "se_area_um2": se_area * PIXEL2_TO_UM2,
                "mean_max_feret_diameter": mean_feret,
                "mean_max_feret_diameter_um": mean_feret * PIXEL_TO_UM,
                "std_max_feret_diameter": std_feret,
                "std_max_feret_diameter_um": std_feret * PIXEL_TO_UM,
                "se_max_feret_diameter": se_feret,
                "se_max_feret_diameter_um": se_feret * PIXEL_TO_UM,
            }
        )

# Create DataFrames and save to CSV
boxplot_df = pd.DataFrame(boxplot_records)
timeline_df = pd.DataFrame(timeline_records)

boxplot_df.to_csv("data_for_boxplots.csv", index=False)
timeline_df.to_csv("data_for_timeline_charts.csv", index=False)

loginfo(f"\nSaved: data_for_boxplots.csv ({len(boxplot_df)} objects)")
loginfo(
    f"Saved: data_for_timeline_charts.csv ({len(timeline_df)} frame-group combinations)"
)
loginfo("Done!")
