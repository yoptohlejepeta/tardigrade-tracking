import marimo

__generated_with = "0.15.5"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd

    group = "CytochalcasinD"
    video = "CD5.tuns.mkv"
    data = pd.read_csv(f"processed_videos/{group}/{video}/objects_data.csv")
    # data = pd.read_csv(f"processed_video/")
    data.head()
    return data, video


@app.cell
def _(data):
    clusters = data.groupby("frame")["label"].nunique().reset_index(name="num_labels")
    clusters
    return (clusters,)


@app.cell
def _():
    from pygal.style import DefaultStyle

    custom_style = DefaultStyle
    custom_style.font_family = "MartianMono NF"
    custom_style.title_font_size = 25.0
    custom_style.label_font_size = 20
    custom_style.major_label_font_size = 23
    custom_style.stroke_width = 2
    return (custom_style,)


@app.cell
def _(clusters, custom_style, video):
    import pygal

    cluster_counts = clusters['num_labels'].value_counts().sort_index()

    bar_chart = pygal.Bar(
        title=video,
        x_title='Number of objects',
        y_title='Number of Frames',
        show_legend=False,
        style=custom_style,
    )

    bar_chart.add('Frames', [
        {'value': count, 'label': str(count)} for count in cluster_counts.tolist()
    ])

    bar_chart.x_labels = cluster_counts.index.astype(str).tolist()

    bar_chart.render_to_file(f'graphs/obj_count/{video}_hist.svg')
    return (pygal,)


@app.cell
def _(clusters, custom_style, pygal, video):
    line_chart = pygal.Line(
        title=video,
        x_title='Frame',
        y_title='Number of objects',
        style=custom_style,
        show_dots=False,
        truncate_label=-1,
        x_labels = [str(frame) if frame % 1000 == 0 else '' for frame in clusters['frame']],
        width=1600,
        y_labels=clusters["num_labels"].unique().tolist(),
        show_legend=False,
    )

    line_chart.add("Frames", clusters["num_labels"].tolist())

    line_chart.render_to_file(f"graphs/obj_count/{video}_line.svg")
    # line_chart.render_to_png("line.png")
    return


@app.cell
def _(data):
    mean_area = data.groupby("frame")["area"].mean()

    mean_area
    return


@app.cell
def _(custom_style, data, pygal, video):

    metrics = ['area', 'eccentricity','extent', 'major_axis_length', 'max_feret_diameter', 'minor_axis_length', 'perimeter', 'solidity', 'sphericity']

    for metric in metrics:
        grouped_stats = data.groupby('frame')[metric].agg(['min', 'max', 'mean']).reset_index()
        grouped_stats = grouped_stats.dropna(subset=['min', 'max', 'mean'])
        line_chart_metric = pygal.Line(
            style=custom_style,
            x_title='Frame',
            y_title=metric.capitalize(),
            title=f'Min, Max, and Mean ({metric.capitalize()})',
            show_dots=False,
            width=1600,
            truncate_label=-1,
            x_labels = [str(frame) if frame % 1000 == 0 else '' for frame in grouped_stats['frame']]
        )

        line_chart_metric.add(f'Min', grouped_stats['min'].tolist(), )
        line_chart_metric.add(f'Max', grouped_stats['max'].tolist(), )
        line_chart_metric.add(f'Mean', grouped_stats['mean'].tolist(), )

        line_chart_metric.render_to_file(f"graphs/metrics/{video}_{metric}.svg")
    return


if __name__ == "__main__":
    app.run()
