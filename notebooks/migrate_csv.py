import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - box data: prumerna hodnota za kazdy objekt, kazde video
    - timeline data: prumerna hodnota za kazdy frame, za celou skupinu
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    import numpy as np

    subgroup_1 = [
        "T49.tuns",
        "T50.tuns",
        "T51.tuns",
        "T52.tuns",
        "T53.tuns",
        "T54.tuns",
        "T55.tuns",
        "T56.tuns",
        "T57.tuns",
        "T58.tuns",
        "T59.tuns",
        "T60.tuns",
    ]
    paths = Path("T_10um_csv_filtered/").glob("*.csv")

    PIXEL_TO_UM = 3.33
    PIXEL2_TO_UM2 = PIXEL_TO_UM**2

    boxplot_records = []
    timeline_records = []

    dfs = []

    for file in paths:
        video = file.stem
        valid_data = pd.read_csv(file)

        if max(valid_data["frame"]) < 3000:
            valid_data["frame"] = valid_data["frame"] * 2

            def fill_gaps(df):
                current_track_id = df.name 

                df = df.set_index("frame")
                new_frames = range(df.index.min(), df.index.max() + 1)
                df = df.reindex(new_frames).ffill()

                df["track_id"] = current_track_id
                df["track_id"] = df["track_id"].astype(int)

                return df.reset_index()

            valid_data = valid_data.groupby("track_id", group_keys=False).apply(fill_gaps)

        group = "Taxol"
        subgroup = "10"

        valid_data["subgroup"] = subgroup
        valid_data["dx"] = valid_data.groupby("track_id")["centroid_x"].diff().fillna(0)
        valid_data["dy"] = valid_data.groupby("track_id")["centroid_y"].diff().fillna(0)
        valid_data["displacement"] = np.sqrt(
            valid_data["dx"] ** 2 + valid_data["dy"] ** 2
        )
        valid_data["video"] = video

        for track_id in valid_data["track_id"].unique():
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

        dfs.append(valid_data)

    df = pd.concat(dfs, ignore_index=True)

    for subgroup in df["subgroup"].unique():
        subgroup_data = df[df["subgroup"] == subgroup]
        for frame in df["frame"].unique():
            frame_data = subgroup_data[subgroup_data["frame"] == frame]

            mean_disp = frame_data["displacement"].mean()
            std_disp = frame_data["displacement"].std()
            se_disp = std_disp / np.sqrt(len(frame_data))

            mean_area = frame_data["area"].mean()
            std_area = frame_data["area"].std()
            se_area = std_area / np.sqrt(len(frame_data))

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
    return boxplot_records, mo, pd, timeline_records


@app.cell
def _(boxplot_records, pd, timeline_records):
    boxplot_df = pd.DataFrame(boxplot_records)
    timeline_df = pd.DataFrame(timeline_records)
    return boxplot_df, timeline_df


@app.cell
def _(boxplot_df):
    boxplot_df[boxplot_df["group"] == "Taxol"]["subgroup"].unique()
    return


@app.cell
def _(boxplot_df, pd):
    old_box_data = pd.read_csv("old_csv/data_for_boxplots_J.csv", dtype={"subgroup": str})

    new_box_data = pd.concat([old_box_data, boxplot_df], ignore_index=True)
    # new_box_data[new_box_data["group"] == "Taxol"]["subgroup"].unique()
    new_box_data = new_box_data.drop(new_box_data[(new_box_data["group"] == "Taxol") & (new_box_data["subgroup"] == "100")].index)
    return (new_box_data,)


@app.cell
def _(new_box_data):
    new_box_data[(new_box_data["group"] == "Taxol") & (new_box_data["subgroup"] == "10")]["track_id"].unique()
    return


@app.cell
def _(pd, timeline_df):
    old_line_data = pd.read_csv("old_csv/data_for_timeline_charts_J.csv", dtype={"subgroup": str})

    new_line_data = pd.concat([old_line_data, timeline_df], ignore_index=True)
    new_line_data = new_line_data.drop(new_line_data[(new_line_data["group"] == "Taxol") & (new_line_data["subgroup"] == "100")].index)
    return (new_line_data,)


@app.cell
def _(new_line_data):
    new_line_data["time_seconds"] = new_line_data["frame"] / 60
    return


@app.cell
def _(new_box_data, new_line_data):
    new_box_data.to_csv("data_for_boxplots.csv", index=False)
    new_line_data.to_csv("data_for_timeline_charts.csv", index=False)
    return


@app.cell
def _(new_line_data):
    new_line_data["group_subgroup"] = new_line_data["group"] + " - " + new_line_data["subgroup"]
    new_line_data["group_subgroup"].unique()
    return


if __name__ == "__main__":
    app.run()
