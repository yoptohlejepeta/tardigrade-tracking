import marimo

__generated_with = "0.15.5"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd

    boxplot_data = pd.read_csv("data_for_boxplots.csv")
    return boxplot_data, pd


@app.cell
def _(boxplot_data):
    boxplot_data.loc[boxplot_data["group"] == "Taxol"]["video"].unique()
    return


@app.cell
def _(boxplot_data):
    un_ids = boxplot_data.loc[boxplot_data["video"] == "T25.tuns"]["track_id"].unique()

    sorted(un_ids)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Control

    - C11: 10, 32, 33, 6 | 20
    - C13: | 7, 10, 11
    - C25: | 20, 27
    - C4: 4,
    - C33: 18 | 10, 6, 

    ## CytochalasinD

    - CD2: | 17
    - CD3: | 2,
    - CD4: | 4, 13, 18,
    - CD19: 18 | 7, 9, 16
    - CD26: | 3 
    - CD9: 1

    ## Taxol

    - T25: | 28

    ## Nocodazole

    - N8: 1
    - N33: 10, 12
    - N30: 30
    - N32: | 13
    - N34: | 13, 22, 14, 17, 19
    - N35: | 2, 3
    - N39: | 23
    """
    )
    return


@app.cell
def _(pd):
    def _():
        # remove objects with noted ids from said videos

        boxplot_data = pd.read_csv("data_for_boxplots.csv")
        boxplot_data = boxplot_data[~((boxplot_data["video"] == "C11.tuns") & (boxplot_data["track_id"].isin([10, 32, 33])))]

        boxplot_data = boxplot_data[~((boxplot_data["video"] == "C4.tuns") & (boxplot_data["track_id"].isin([4])))]

        boxplot_data = boxplot_data[~((boxplot_data["video"] == "C33.tuns") & (boxplot_data["track_id"].isin([18])))]

        boxplot_data = boxplot_data[~((boxplot_data["video"] == "CD19.tuns") & (boxplot_data["track_id"].isin([18])))]

        boxplot_data = boxplot_data[~((boxplot_data["video"] == "CD9.tuns") & (boxplot_data["track_id"].isin([1])))]

        boxplot_data = boxplot_data[~((boxplot_data["video"] == "N33.tuns") & (boxplot_data["track_id"].isin([10, 12])))]
        return boxplot_data.to_csv("data_for_boxplots_cleaned.csv", index=False)


    _()
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
