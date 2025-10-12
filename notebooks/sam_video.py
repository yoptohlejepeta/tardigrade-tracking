import marimo

__generated_with = "0.15.5"
app = marimo.App(width="full")


@app.cell
def _():
    from ultralytics.models.sam import SAM2VideoPredictor

    # Create SAM2VideoPredictor
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt")
    predictor = SAM2VideoPredictor(overrides=overrides)
    return (predictor,)


@app.cell
def _(predictor):
    video_path = "data/C2.tuns.mkv"
    results = predictor(source=video_path, points=[604, 390], labels=[1])
    return


if __name__ == "__main__":
    app.run()
