import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Zpracování snímků želvušek
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Původní snímek
    """)
    return


@app.cell
def _(mo):
    import imageio.v3 as iio
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage as ndi

    image_path = mo.ui.file_browser(multiple=False)
    image_path
    return iio, image_path, ndi, np, plt


@app.cell
def _(iio, image_path, mo):
    image = iio.imread(image_path.path(index=0), index=0)
    height, width = image.shape[:2]
    image[int(height * 0.8) :, int(width * 0.7) :] = 0

    mo.image(image)
    return (image,)


@app.cell
def _(image):
    print(image.shape)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Unsharp masking
    """)
    return


@app.cell
def _(image, mo):
    from skimage.filters import unsharp_mask, gaussian

    unsharped = unsharp_mask(image, amount=3)

    mo.image(unsharped)
    return gaussian, unsharped


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

    mo.image(gray)
    return (gray,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Gaussův filtr
    """)
    return


@app.cell
def _(gaussian, gray, plt):
    gray_gauss = gaussian(gray, sigma=3)

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(gray_gauss, cmap="gray")
    return (gray_gauss,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Binarizace
    """)
    return


@app.cell
def _(gray_gauss, plt):
    from skimage.filters import threshold_otsu


    otsu_image = gray_gauss > threshold_otsu(gray_gauss)

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(otsu_image)
    return (otsu_image,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Morfologické operátory
    """)
    return


@app.cell
def _(otsu_image, plt):
    from skimage.morphology import remove_small_objects, opening, disk, erosion, dilation, remove_small_holes, closing

    removed = opening(
        otsu_image,
        disk(10),
    )

    removed = remove_small_objects(removed, min_size=2000)

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(removed)
    return disk, removed


@app.cell
def _(mo):
    mo.md(r"""
    ## Watershed
    """)
    return


@app.cell
def _(disk, ndi, np, plt, removed):
    from skimage.segmentation import watershed, clear_border
    from skimage.feature import peak_local_max
    from scipy import ndimage

    final_image = removed

    distance = ndi.distance_transform_edt(removed)
    coords = peak_local_max(
        distance,
        labels=removed,
        min_distance=20,
        # threshold_rel=0.25,
        footprint=disk(40),
        exclude_border=True,
    )
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(input=mask)  # pyright: ignore[reportGeneralTypeIssues]
    labels = watershed(
        -distance, markers, mask=removed, connectivity=2
    )  # pyright: ignore

    plt.figure(figsize=(50, 50))
    plt.axis("off")
    plt.imshow(labels)
    plt.show()
    return coords, labels


@app.cell
def _(coords, image, labels, np, plt):
    from PIL import Image, ImageDraw, ImageFont
    from skimage.segmentation import find_boundaries
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

        boundaries = find_boundaries(labels, mode="outer")
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
def _(image, labels, mo, plt):
    from skimage.color import label2rgb

    plt.imshow(image, cmap="gray")
    label_overlay = label2rgb(labels, image=image, bg_label=0, alpha=0.3)

    mo.image(label_overlay)
    return


if __name__ == "__main__":
    app.run()
