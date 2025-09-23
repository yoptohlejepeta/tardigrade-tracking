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

    image = iio.imread("data/C3.tuns.tif")
    image = image[:, 250:1500 ,:]
    plt.imshow(image)
    return iio, image, mo, ndi, np, plt


@app.cell
def _(mo):
    mo.md(r"""## Grayscale""")
    return


@app.cell
def _(image, plt):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]

    plt.imshow(1/3*r + 1/3*g + 1/3*b,cmap="grey")
    return b, g, r


@app.cell
def _(mo):
    mo.md(r"""## Binarizace""")
    return


@app.cell
def _(b, g, plt, r):
    from skimage.filters import threshold_otsu

    img_part = 1/3*r + 1/3*g + 1/3*b
    otsu_image = img_part > threshold_otsu(img_part)

    plt.imshow(otsu_image)
    return (otsu_image,)


@app.cell
def _(mo):
    mo.md(r"""## Dilatace""")
    return


@app.cell
def _(np, otsu_image, plt):
    from skimage.morphology import closing, dilation

    for close in range(5):
        closed = dilation(otsu_image, footprint=np.ones((5,5)))

    plt.imshow(closed)
    return (closed,)


@app.cell
def _(mo):
    mo.md(r"""## Rozložení velikosti objektů a odstranění malých objektů""")
    return


@app.cell
def _(closed, ndi, np, plt):
    closed_labels, num_objects = ndi.label(closed)

    closed_unique_labels = np.unique(closed_labels)
    closed_unique_labels = closed_unique_labels[closed_unique_labels != 0]

    object_sizes = np.bincount(closed_labels.ravel())[closed_unique_labels]

    plt.hist(object_sizes, bins=20, edgecolor='black')
    plt.xlabel('size')
    plt.show()
    return


@app.cell
def _(closed, plt):
    from skimage.morphology import remove_small_objects

    removed = remove_small_objects(closed, min_size=2500)

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

    final_image = removed

    distance = ndi.distance_transform_edt(final_image)
    coords = peak_local_max(distance, labels=final_image, min_distance=30)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=final_image, connectivity=2)

    plt.imshow(labels)
    return (labels,)


@app.cell
def _(image, labels, ndi, np, plt):
    # unique_labels = np.unique(labels)
    # unique_labels = unique_labels[unique_labels != 0]
    # centroids = []
    # for label in unique_labels:
    #     centroid = ndi.center_of_mass(labels == label)
    #     centroids.append(centroid)

    # centroids = np.array(centroids)

    # plt.imshow(labels)
    # plt.scatter(centroids[:, 1], centroids[:, 0], c='red', s=50, marker='o', label='Centroids')
    # plt.show()

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]  # Exclude background
    centroids = []
    for label in unique_labels:
        centroid = ndi.center_of_mass(labels == label)
        centroids.append(centroid)
    centroids = np.array(centroids)

    plt.imshow(image)

    for label in unique_labels:
        label_mask = labels == label
        colored_mask = np.zeros((*label_mask.shape, 4))
        color = np.random.rand(3)
        colored_mask[label_mask, :3] = color
        colored_mask[label_mask, 3] = 0.5
        plt.imshow(colored_mask)

    plt.scatter(centroids[:, 1], centroids[:, 0], c='red', s=20, marker='o')
    plt.show()
    return


@app.cell
def _(iio):
    images = iio.imiter("./data/C2.tuns.mkv")

    count = 0
    for _ in images:
        count += 1

    print(count)
    return


if __name__ == "__main__":
    app.run()
