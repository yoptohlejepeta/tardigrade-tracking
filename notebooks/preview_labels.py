import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import imageio.v3 as iio
    from PIL import Image
    return Image, iio, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    ## Data folder
    """)
    return


@app.cell
def _(mo):
    data_path = mo.ui.file_browser(selection_mode="directory", multiple=False)
    data_path
    return (data_path,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Label path
    """)
    return


@app.cell
def _(mo):
    labels_path = mo.ui.file_browser(selection_mode="directory", multiple=False)
    labels_path
    return (labels_path,)


@app.cell
def _(labels_path):
    print(labels_path.name())
    return


@app.cell
def _(labels_path, np):
    if labels_path.value:
        labels = labels_path.path().glob("*")
        label_ims = {}
        for label in labels:
            frame_num = int(label.stem.split("_")[1])
            label_ims[frame_num] = np.load(label)
    return (label_ims,)


@app.cell
def _(Image, data_path, iio, label_ims, labels_path, mo, np):
    import matplotlib.pyplot as plt

    colors = [
        [255, 0, 0],   # Red
        [0, 255, 0],   # Green
        [0, 0, 255],   # Blue
        [255, 255, 0], # Yellow
        [255, 0, 255], # Magenta
        [0, 255, 255]  # Cyan
    ]

    processed_frames = []
    alpha = 0.5
    video_path = data_path.path() / f"{labels_path.name()}.mkv"

    for num, mask in label_ims.items():
        frame = iio.imread(video_path, index=num)
    
        overlay = np.zeros_like(frame)
    
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids != 0]
    
        for obj_id in object_ids:
            color = colors[int(obj_id) % len(colors)]
            overlay[mask == obj_id] = color

        blended = (frame * (1 - alpha) + overlay * alpha).astype(np.uint8)
    
        processed_frames.append(
            mo.image(
                src=Image.fromarray(blended), 
                caption=f"Frame: {num}",
                width=800
            )
        )

    carousel = mo.carousel(processed_frames)
    carousel
    return


if __name__ == "__main__":
    app.run()
