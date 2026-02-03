import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import imageio.v3 as iio
    from PIL import Image, ImageDraw, ImageFont
    return Image, ImageDraw, ImageFont, iio, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    ## Label path
    """)
    return


@app.cell
def _(mo):
    # slozka s obrazky a labely
    labels_path = mo.ui.file_browser(selection_mode="directory", multiple=False)
    labels_path
    return (labels_path,)


@app.cell
def _(labels_path, np):
    if labels_path.value:
        labels = labels_path.path().glob("*.npy")
        label_ims = {}
        for label in labels:
            label_ims[label.stem] = np.load(label)
    return (label_ims,)


@app.cell
def _(Image, ImageDraw, ImageFont, iio, label_ims, labels_path, mo, np):
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

    for num, mask in label_ims.items():
        frame = iio.imread(labels_path.path() / f"{num}.png")
        overlay = np.zeros_like(frame)
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids != 0]

        for obj_id in object_ids:
            color = colors[int(obj_id) % len(colors)]
            overlay[mask == obj_id] = color

        blended = (frame * (1 - alpha) + overlay * alpha).astype(np.uint8)
    
        img = Image.fromarray(blended)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=24)
    
        for obj_id in object_ids:
            y, x = np.where(mask == obj_id)
            if len(y) > 0:
                cx, cy = int(np.mean(x)), int(np.mean(y))
                for adj_x, adj_y in [(-2,-2), (-2,2), (2,-2), (2,2), (-2,0), (2,0), (0,-2), (0,2)]:
                    draw.text((cx+adj_x, cy+adj_y), str(obj_id), fill=(0, 0, 0), font=font)
                draw.text((cx, cy), str(obj_id), fill=(255, 255, 0), font=font)
    
        processed_frames.append(
            mo.image(
                src=img, 
                caption=f"Frame: {num}",
                width=800
            )
        )

    carousel = mo.carousel(processed_frames)
    carousel
    return


if __name__ == "__main__":
    app.run()
