import marimo

__generated_with = "0.15.5"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd

    video = "C4.tuns.mkv"
    data = pd.read_csv(f"images_test/{video}/objects_data.csv")
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
    return (custom_style,)


@app.cell
def _(clusters, custom_style, video):
    import pygal

    cluster_counts = clusters['num_labels'].value_counts().sort_index()

    bar_chart = pygal.Bar(
        title=video,
        x_title='N objects',
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
        y_title='Number of Unique Clusters',
        style=custom_style,
        show_dots=False,
        truncate_label=-1,
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
    return (mean_area,)


@app.cell
def _(custom_style, mean_area, pygal, video):
    line_chart_area = pygal.Line(
        title=video,
        x_title='Frame',
        y_title='Mean area (pixels)',
        style=custom_style,
        show_dots=False,
        truncate_label=-1,
        width=1600,
        # y_labels=clusters["num_labels"].unique().tolist(),
        show_legend=False,
    )

    line_chart_area.add("Frames", mean_area.tolist())

    # line_chart.render_to_file(f"graphs/obj_count/{video}_line.svg")
    return


if __name__ == "__main__":
    app.run()
