import marimo

__generated_with = "0.15.5"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""# Zpracování snímků želvušek""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Původní snímek""")
    return


@app.cell
def _():
    import marimo as mo
    import imageio.v3 as iio
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage as ndi

    plt.axis("off")

    image = iio.imread("data/C2.tuns.tif")
    image = image[:, 250:1500 ,:]
    plt.imshow(image)
    return image, mo, ndi, np, plt


@app.cell
def _(image):
    print(image.shape)
    return


@app.cell
def _(mo):
    mo.md(r"""## Unsharp masking""")
    return


@app.cell
def _(image, plt):
    from skimage.filters import unsharp_mask

    unsharped = unsharp_mask(image, amount=3)

    plt.axis("off")
    plt.imshow(unsharped)
    return (unsharped,)


@app.cell
def _(mo):
    mo.md(r"""## Grayscale""")
    return


@app.cell
def _(plt, unsharped):
    prep = unsharped

    r, g, b = prep[:,:,0], prep[:,:,1], prep[:,:,2]
    gray = 1/3*r + 1/3*g + 1/3*b

    plt.axis("off")
    plt.imshow(gray,cmap="gray")
    return (gray,)


@app.cell
def _(mo):
    mo.md(r"""## Gaussův filtr""")
    return


@app.cell
def _(gray, plt):
    from skimage.filters import gaussian

    gray_gauss = gaussian(gray, sigma=3)

    plt.axis("off")
    plt.imshow(gray_gauss, cmap="gray")
    return (gray_gauss,)


@app.cell
def _(mo):
    mo.md(r"""## Binarizace""")
    return


@app.cell
def _(gray_gauss, plt):
    from skimage.filters import threshold_otsu

    otsu_image = gray_gauss > threshold_otsu(gray_gauss)

    plt.axis("off")
    plt.imshow(otsu_image)
    return (otsu_image,)


@app.cell
def _(mo):
    mo.md(r"""## Morfologické operátory""")
    return


@app.cell
def _(np, otsu_image, plt):
    from skimage.morphology import remove_small_objects, opening

    removed = remove_small_objects(otsu_image, min_size=2000)
    removed = opening(removed, footprint=np.ones((10, 10)))

    plt.axis("off")
    plt.imshow(removed)
    return (removed,)


@app.cell
def _(mo):
    mo.md(r"""## Watershed""")
    return


@app.cell
def _(ndi, np, plt, removed):
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from skimage.morphology import h_maxima, local_maxima, disk
    from scipy import ndimage

    final_image = removed

    distance = ndi.distance_transform_edt(removed)
    coords = peak_local_max(
        distance, labels=removed, min_distance=30, threshold_rel=0.5, footprint=disk(35)
    )
    mask = np.zeros(distance.shape, dtype=bool)  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(input=mask)  # pyright: ignore[reportGeneralTypeIssues]
    labels = watershed(-distance, markers, mask=removed, connectivity=1)  # pyright: ignore

    plt.axis("off")
    plt.imshow(labels)
    return coords, labels


@app.cell
def _(coords, image, labels, plt):
    from skimage.color import label2rgb

    plt.imshow(image, cmap='gray')
    label_overlay = label2rgb(labels, image=image, bg_label=0, alpha=0.3)
    plt.imshow(label_overlay)
    plt.plot(coords[:, 1], coords[:, 0], 'o', markersize=3,color="red", label='Centroids')
    plt.axis('off')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
