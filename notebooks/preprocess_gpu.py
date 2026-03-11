import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    import imageio.v3 as iio
    import matplotlib.pyplot as plt
    import numpy as np

    import cupy as cp

    image_path = mo.ui.file_browser(multiple=False)
    image_path
    return cp, iio, image_path, np, plt


@app.cell
def _(cp, iio, image_path, mo):
    image = iio.imread(image_path.path(index=0), index=0)
    height, width = image.shape[:2]
    image[int(height * 0.8) :, int(width * 0.7) :] = 0
    image_gpu = cp.asarray(image)


    mo.image(image_gpu.get())
    return image, image_gpu


@app.cell
def _(mo):
    mo.md(r"""
    ## Unsharp masking
    """)
    return


@app.cell
def _(image_gpu):
    from cucim.skimage.filters import unsharp_mask

    unsharped = unsharp_mask(image_gpu, amount=3, radius=1, channel_axis=1)
    return (unsharped,)


@app.cell
def _(mo, unsharped):
    mo.image(unsharped.get())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Grayscale
    """)
    return


@app.cell
def _(mo, unsharped):
    prep = unsharped

    r, g, b = prep[:, :, 0], prep[:, :, 1], prep[:, :, 2]
    gray = 1 / 3 * r + 1 / 3 * g + 1 / 3 * b

    mo.image(gray.get())
    return (gray,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Gaussův filtr
    """)
    return


@app.cell
def _(gray, mo):
    from cucim.skimage.filters import gaussian

    gray_gauss = gaussian(gray, sigma=3)

    mo.image(gray_gauss.get())
    return (gray_gauss,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Binarizace
    """)
    return


@app.cell
def _(gray_gauss, plt):
    from cucim.skimage.filters import threshold_otsu


    otsu_image = gray_gauss > threshold_otsu(gray_gauss)

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(otsu_image.get())
    return (otsu_image,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Morfologické operátory
    """)
    return


@app.cell
def _(otsu_image):
    from cucim.skimage.morphology import remove_small_objects, opening, disk, erosion, dilation, remove_small_holes, closing

    removed = opening(
        otsu_image,
        disk(10),
    )

    removed = remove_small_objects(removed, min_size=2000)
    return disk, removed


@app.cell
def _(plt, removed):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(removed.get())
    return


@app.cell
def _(cp, disk, np, removed):
    from skimage.segmentation import watershed, clear_border
    from cucim.skimage.feature import peak_local_max
    # from scipy import ndimage
    from scipy.ndimage import distance_transform_edt, label

    final_image = removed

    distance = distance_transform_edt(removed.get())
    coords = peak_local_max(
        cp.asarray(distance),
        labels=removed,
        min_distance=20,
        # threshold_rel=0.25,
        footprint=disk(40),
        exclude_border=True,
    )
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.get().T)] = True
    markers, _ = label(input=mask)  # pyright: ignore[reportGeneralTypeIssues]
    labels = watershed(
        -distance,
        markers,
        mask=removed.get(), connectivity=2
    )  # pyright: ignore
    return coords, labels


@app.cell
def _(labels, plt):
    plt.figure(figsize=(50, 50))
    plt.axis("off")
    plt.imshow(labels)
    plt.show()
    return


@app.cell
def _(coords, cp, image, labels, np, plt):
    from PIL import Image, ImageDraw, ImageFont
    from cucim.skimage.segmentation import find_boundaries
    # from skimage.morphology import dilation


    def display_image_with_boundaries(
        image: np.ndarray,
        labels: np.ndarray,
        coords: np.ndarray,  # centroids as (y, x) coordinates
        track_ids: np.ndarray = None,  # optional track IDs
        boundary_thickness: int = 2,
    ) -> None:
        if image.ndim == 2:  # grayscale
            img = Image.fromarray(image).convert("RGB")
        else:
            img = Image.fromarray(image.astype(np.uint8)).convert("RGB")

        draw = ImageDraw.Draw(img)

        boundaries = find_boundaries(cp.asarray(labels), mode="outer")
        # boundaries = dilation(boundaries, disk(boundary_thickness))
        boundary_coords = np.argwhere(boundaries)

        for y, x in boundary_coords:
            draw.point((x, y), fill=(255, 0, 0))

        font = ImageFont.load_default(size=30)

        if len(coords) > 0:
            for i, (cy, cx) in enumerate(coords):
                draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(255, 0, 0))

                if track_ids is not None:
                    label_text = str(int(track_ids[i]))
                    bbox = draw.textbbox((cx, cy), label_text, font=font)
                    text_height = bbox[3] - bbox[1]

                    text_x = cx + 8
                    text_y = cy - text_height // 2

                    # draw.text(
                    #     (text_x, text_y), label_text, fill=(0, 255, 0), font=font
                    # )

        plt.figure(figsize=(50, 50))
        plt.axis("off")
        plt.imshow(np.array(img))
        plt.show()


    display_image_with_boundaries(
        image, labels, coords, track_ids=np.arange(1, len(coords) + 1)
    )
    return


@app.cell
def _(cp, image_gpu, labels):
    from cucim.skimage.color import label2rgb

    # plt.imshow(image, cmap="gray")
    label_overlay = label2rgb(cp.asarray(labels), image=image_gpu, bg_label=0, alpha=0.3)
    return (label_overlay,)


@app.cell
def _(label_overlay, mo):
    mo.image(label_overlay.get())
    return


if __name__ == "__main__":
    app.run()
