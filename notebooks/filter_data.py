import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import pandas as pd

    n = 60

    remove_objs = []

    data = pd.read_csv(f"T_10um_csv/T{n}.tuns.csv")
    data = data[~data["track_id"].isin(remove_objs)]

    total_frames = data['frame'].nunique()
    track_frame_counts = data.groupby('track_id')['frame'].nunique()
    consistent_tracks = track_frame_counts[track_frame_counts == total_frames].index

    data = data[data['track_id'].isin(consistent_tracks)]
    return data, n


@app.cell
def _(data):
    data.track_id.unique()
    return


@app.cell
def _(data, n):
    data.to_csv(f"T_10um_csv_filtered/T{n}.tuns.csv", index=False)
    return


if __name__ == "__main__":
    app.run()
