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

    image = iio.imread("data/C4.tuns.tif")
    image = image[:, 250:1500 ,:]
    plt.imshow(image)
    return image, mo, ndi, np, plt


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
def _(gray, plt):
    from skimage.filters import gaussian

    gray_gauss = gaussian(gray, sigma=2)

    plt.imshow(gray_gauss, cmap="gray")
    return (gray_gauss,)


@app.cell
def _(mo):
    mo.md(r"""## Binarizace""")
    return


@app.cell
def _(gray, plt):
    from skimage.filters import threshold_otsu

    otsu_image = gray > threshold_otsu(gray)

    plt.axis("off")
    plt.imshow(otsu_image)
    return otsu_image, threshold_otsu


@app.cell
def _(gray_gauss, plt, threshold_otsu):
    otsu_gray = gray_gauss > threshold_otsu(gray_gauss)

    plt.imshow(otsu_gray)
    return (otsu_gray,)


@app.cell
def _(np, opening, otsu_gray, plt, remove_small_objects):
    # eros = erosion(otsu_image, footprint=np.ones((3,3)))
    removed_gaus = remove_small_objects(otsu_gray, min_size=2000)
    # removed_gaus = opening(removed_gaus, footprint=np.ones((5,5)))
    removed_gaus = opening(removed_gaus, footprint=np.ones((10,10)))
    # removed_gaus = closing(removed_gaus, footprint=np.ones((5,5)))
    # removed = closing(removed, footprint=disk(3))

    plt.axis("off")
    plt.imshow(removed_gaus)
    return (removed_gaus,)


@app.cell
def _(ndi, np, peak_local_max, plt, removed_gaus, watershed):
    final_image2 = removed_gaus

    distance2 = ndi.distance_transform_edt(final_image2)
    coords2 = peak_local_max(distance2, labels=final_image2, min_distance=35, threshold_rel=0.6)
    mask2 = np.zeros(distance2.shape, dtype=bool)
    mask2[tuple(coords2.T)] = True
    markers2, _ = ndi.label(input=mask2)
    labels2 = watershed(-distance2, markers2, mask=final_image2, connectivity=1)

    plt.axis("off")
    plt.imshow(labels2)
    return (labels2,)


@app.cell
def _(mo):
    mo.md(r"""## Rozložení velikosti objektů a odstranění malých objektů""")
    return


@app.cell
def _(ndi, np, otsu_image, plt):
    closed_labels, num_objects = ndi.label(otsu_image)

    closed_unique_labels = np.unique(closed_labels)
    closed_unique_labels = closed_unique_labels[closed_unique_labels != 0]

    object_sizes = np.bincount(closed_labels.ravel())[closed_unique_labels]

    plt.hist(object_sizes, bins=100, edgecolor='black')
    plt.xlabel('size')
    plt.show()
    return


@app.cell
def _(otsu_image, plt):
    from skimage.morphology import remove_small_objects, disk
    from skimage.morphology import closing, dilation, erosion, opening

    # eros = erosion(otsu_image, footprint=np.ones((3,3)))
    removed = remove_small_objects(otsu_image, min_size=2000)
    removed = closing(removed)
    # removed = opening(removed, footprint=np.ones((30,30)))
    # removed = closing(removed, footprint=disk(3))

    plt.axis("off")
    plt.imshow(removed)
    return disk, opening, remove_small_objects, removed


@app.cell
def _(mo):
    mo.md(r"""## Watershed""")
    return


@app.cell
def _(disk, ndi, np, plt, removed):
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from skimage.morphology import h_maxima, local_maxima
    from scipy import ndimage

    final_image = removed

    distance = ndi.distance_transform_edt(final_image)
    coords = peak_local_max(distance, labels=final_image, min_distance=1, footprint=disk(45))
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(input=mask)
    labels = watershed(-distance, markers, mask=final_image, connectivity=1)

    plt.axis("off")
    plt.imshow(labels)
    return peak_local_max, watershed


@app.cell
def _(image, labels2, ndi, np, plt):
    unique_labels = np.unique(labels2)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background
    centroids = []
    for label in unique_labels:
        centroid = ndi.center_of_mass(labels2 == label)
        centroids.append(centroid)
    centroids = np.array(centroids)

    plt.imshow(image)

    for label in unique_labels:
        label_mask = labels2 == label
        colored_mask = np.zeros((*label_mask.shape, 4))
        color = np.random.rand(3)
        colored_mask[label_mask, :3] = color
        colored_mask[label_mask, 3] = 0.5
        plt.imshow(colored_mask)

    plt.scatter(centroids[:, 1], centroids[:, 0], c='red', s=20, marker='o')
    plt.axis("off")
    plt.show()
    return


@app.cell
def _():
    # images = iio.imiter("./data/C2.tuns.mkv")

    # count = 0
    # for _ in images:
    #     count += 1

    # print(count)
    return


if __name__ == "__main__":
    app.run()
