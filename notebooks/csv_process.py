import marimo

__generated_with = "0.15.5"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import numpy as np

    # video = "C11.tuns"
    # df = pd.read_csv(f"./csvs/Control/{video}/{video}.csv")
    group = "Control"
    subgroup = "0.5"
    video_ids = [2]

    dfs = []

    for video_id in video_ids:
        video = f"C{video_id}.tuns"
        df_video = pd.read_csv(f"./csvs/{group}/{video}/{video}.csv")
        track_ids_in_all_frames = df_video["track_id"].value_counts()
        num_frames = df_video["frame"].nunique()

        track_ids_in_all_frames = track_ids_in_all_frames[
            track_ids_in_all_frames == num_frames
        ].index.tolist()
        valid_data = df_video[df_video["track_id"].isin(track_ids_in_all_frames)]
        valid_data = valid_data.sort_values(by=["track_id", "frame"])
        valid_data["dx"] = (
            valid_data.groupby("track_id")["centroid_x"].diff().fillna(0)
        )
        valid_data["dy"] = (
            valid_data.groupby("track_id")["centroid_y"].diff().fillna(0)
        )
        valid_data["displacement"] = np.sqrt(
            valid_data["dx"] ** 2 + valid_data["dy"] ** 2
        )

        valid_data["video"] = video

        dfs.append(valid_data)

    df = pd.concat(dfs, ignore_index=True)
    df
    return df, group, np, subgroup, valid_data


@app.cell
def _(valid_data):
    # show row with max displacement
    valid_data.loc[valid_data["displacement"].idxmax()][
        ["track_id", "frame", "displacement"]
    ]
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    pridat displacement, mean square displacement, area, zmena tvaru (feret)
    aktivni/neaktivni

    1. selekce snimku
    2. jaka videa dohromady, seskupit podle OPUS.xlsx

    jak spocitat mobility index (nebo podobnou metriku)


    pozor na skupinu

    objekty pres vsehna videa v dane skupine

    1. vyfiltrovat objekty u kterých se mění indexy
    2. analýza mobility - displacement (zkusit zatim jen tunsy, celkovy displacement)



    suma delta r na druhou

    stredni displacement

    potom area (stejny postup)
    """
    )
    return


@app.cell
def _():
    from pygal.style import DefaultStyle

    custom_style = DefaultStyle
    custom_style.title_font_size = 25.0
    custom_style.label_font_size = 20
    custom_style.major_label_font_size = 23
    custom_style.stroke_width = 2
    return (custom_style,)


@app.cell
def _(custom_style, df, group, subgroup):
    # displacement chart
    import pygal
    line_chart = pygal.Line(
        x_label_rotation=20,
        style=custom_style,
        width=2000,
        show_dots=False,
    )
    line_chart.title = f"Displacement over Frames ({group} - {subgroup})"
    mean_displacement_per_frame = df.groupby("frame")["displacement"].mean()

    line_chart.x_labels = [str(i) if i % 1000 == 0 else '' for i in mean_displacement_per_frame.index]

    line_chart.add("Mean Displacement", mean_displacement_per_frame.values)
    line_chart.render_to_png(f"displacement_{group}_{subgroup}.png")
    return (pygal,)


@app.cell
def _(df, group, np, subgroup):
    import matplotlib.pyplot as plt

    # Calculate mean and standard deviation (or standard error) per frame
    stats = df.groupby("frame")["displacement"].agg(['mean', 'std', 'count'])
    stats['se'] = stats['std'] / np.sqrt(stats['count'])  # standard error

    # stats = stats.iloc[::10]

    # For 95% confidence interval
    confidence = 1.96 * stats['se']

    plt.figure(figsize=(16, 6))
    plt.plot(stats.index, stats['mean'], linewidth=2, label='Mean Displacement')
    plt.fill_between(stats.index, 
                      stats['mean'] - confidence, 
                      stats['mean'] + confidence, 
                      alpha=0.3, 
                      label='95% Confidence Interval')

    plt.xlabel('Frame')
    plt.ylabel('Displacement')
    plt.title(f"Displacement over Frames ({group} - {subgroup})")
    plt.xticks(range(0, stats.index.max() + 1, 1000))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"displacement_{group}_{subgroup}_matplotlib.png", dpi=150)
    plt.close()
    return


@app.cell
def _(custom_style, df, group, pygal, subgroup):
    # area chart
    area_chart = pygal.Line(
        x_label_rotation=20,
        style=custom_style,
        width=3200,
        show_dots=False,
    )
    area_chart.title = f"Area over Frames ({group} - {subgroup})"
    mean_area_per_frame = df.groupby("frame")["area"].mean()

    area_chart.x_labels = [str(i) if i % 1000 == 0 else '' for i in mean_area_per_frame.index]

    area_chart.add("Mean Area", mean_area_per_frame.values)
    area_chart.render_to_png(f"area_{group}_{subgroup}.png")
    return


@app.cell
def _(df, group, np, subgroup):
    def _():
        import matplotlib.pyplot as plt

        # Calculate mean and standard deviation (or standard error) per frame
        stats = df.groupby("frame")["area"].agg(['mean', 'std', 'count'])
        stats['se'] = stats['std'] / np.sqrt(stats['count'])  # standard error

        # stats = stats.iloc[::10]

        # For 95% confidence interval
        confidence = 1.96 * stats['se']

        plt.figure(figsize=(16, 6))
        plt.plot(stats.index, stats['mean'], linewidth=2, label='Mean Area')
        plt.fill_between(stats.index, 
                          stats['mean'] - confidence, 
                          stats['mean'] + confidence, 
                          alpha=0.3, 
                          label='95% Confidence Interval')

        plt.xlabel('Frame')
        plt.ylabel('Area')
        plt.title(f"Area over Frames ({group} - {subgroup})")
        plt.xticks(range(0, stats.index.max() + 1, 1000))
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"area_{group}_{subgroup}_matplotlib.png", dpi=150)
        return plt.close()


    _()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
