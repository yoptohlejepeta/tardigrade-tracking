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

    image = iio.imread("data/Taxol/Reh/T1.24Reh.mkv", index=0)
    # image = image[:, 250:1500 ,:]
    height, width = image.shape[:2]
    image[int(height * 0.8):, int(width * 0.7):] = 0
    plt.imshow(image)
    return image, mo, ndi, np, plt


@app.cell
def _(image):
    print(image.shape)
    return


@app.cell
def _(gray_gauss, plt):
    from skimage.exposure import equalize_adapthist

    equalized = equalize_adapthist(gray_gauss)

    plt.axis("off")
    plt.imshow(equalized, cmap="gray")
    return


@app.cell
def _(mo):
    mo.md(r"""## Unsharp masking""")
    return


@app.cell
def _(image, plt):
    from skimage.filters import unsharp_mask, gaussian

    unsharped = unsharp_mask(image, amount=2)

    plt.axis("off")
    plt.imshow(unsharped)
    return gaussian, unsharped


@app.cell
def _(gaussian, plt, unsharped):
    unsh_gaus = gaussian(unsharped, sigma=2)

    plt.axis("off")
    plt.imshow(unsh_gaus)
    return (unsh_gaus,)


@app.cell
def _(mo):
    mo.md(r"""## Grayscale""")
    return


@app.cell
def _(plt, unsh_gaus):
    prep = unsh_gaus

    r, g, b = prep[:,:,0], prep[:,:,1], prep[:,:,2]
    gray = 1/3*r + 1/3*g + 1/3*b

    plt.axis("off")
    plt.imshow(gray,cmap="gray")
    return


@app.cell
def _(mo):
    mo.md(r"""## Gaussův filtr""")
    return


@app.cell
def _(gaussian, plt, unsharped):
    r2, g2, b2 = unsharped[:,:,0], unsharped[:,:,1], unsharped[:,:,2]

    gray_gauss = gaussian(1/3*r2 + 1/3*g2 + 1/3*b2, sigma=3)

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

    binarized = gray_gauss

    otsu_image = binarized > threshold_otsu(binarized)

    plt.axis("off")
    plt.imshow(otsu_image)
    return (otsu_image,)


@app.cell
def _(mo):
    mo.md(r"""## Morfologické operátory""")
    return


@app.cell
def _(otsu_image, plt):
    from skimage.morphology import remove_small_objects, opening, disk

    removed = opening(otsu_image, 
                      footprint=disk(10)
                      # footprint=np.ones((5, 5))
                     )
    removed = remove_small_objects(otsu_image, min_size=2000)

    plt.axis("off")
    plt.imshow(removed)
    return disk, removed


@app.cell
def _(mo):
    mo.md(r"""## Watershed""")
    return


@app.cell
def _(disk, ndi, np, plt, removed):
    from skimage.segmentation import watershed, clear_border
    from skimage.feature import peak_local_max
    from skimage.morphology import h_maxima, local_maxima
    from scipy import ndimage

    final_image = removed

    distance = ndi.distance_transform_edt(removed)
    coords = peak_local_max(
        distance, labels=removed, min_distance=30, threshold_rel=0.25, footprint=disk(45)
    )
    mask = np.zeros(distance.shape, dtype=bool)  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(input=mask)  # pyright: ignore[reportGeneralTypeIssues]
    labels = watershed(-distance, markers, mask=removed, connectivity=2, watershed_line=True)  # pyright: ignore
    labels = clear_border(labels)

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
