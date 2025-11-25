import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


app._unparsable_cell(
    r"""
    import imageio.v3 as iio
    import marimo as mo
    import imageio.v3 as iio
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage as ndi

    first_image = iio.imread(\"data/FC1.0.mkv\", index=0)

    plt.axis('off')
    plt.imshow(first_image

    # binarizace celeho videa
    # zkusit SAM model
    """,
    name="_"
)


@app.cell
def _(first_image, plt):
    r, g, b = first_image[:,:,0], first_image[:,:,1], first_image[:,:,2]
    gray = 1/3*r + 1/3*g + 1/3*b

    plt.axis("off")
    plt.imshow(gray,cmap="gray")
    return (gray,)


@app.cell
def _(gray, plt):
    # rescake intensity
    from skimage.exposure import rescale_intensity

    gray_rescaled = rescale_intensity(gray, in_range=(50, 200), out_range=(0, 255))
    plt.axis("off")
    plt.imshow(gray_rescaled,cmap="gray")
    return (gray_rescaled,)


@app.cell
def _(gray, gray_rescaled, plt):
    #otsu
    from skimage.filters import threshold_otsu

    thresh = threshold_otsu(gray)

    binary = gray_rescaled > thresh

    plt.axis("off")
    plt.imshow(binary,cmap="gray")
    return


@app.cell
def _(gray_rescaled, plt):
    #adaptive thresholding
    from skimage.filters import threshold_local

    block_size = 101
    adaptive_thresh = threshold_local(gray_rescaled, block_size=block_size, offset=0)

    adaptive_binary = gray_rescaled > adaptive_thresh

    plt.axis("off")
    plt.imshow(adaptive_binary,cmap="gray")
    return


@app.cell
def _(gray_rescaled, plt):
    #threesholding overview
    from skimage.filters import try_all_threshold

    fig, ax = try_all_threshold(gray_rescaled, figsize=(10, 8), verbose=False)
    plt.show()
    return


@app.cell
def _(gray_rescaled, plt):
    #yen thrsholding
    from skimage.filters import threshold_yen

    yen_thresh = threshold_yen(gray_rescaled)
    yen_binary = gray_rescaled > yen_thresh

    plt.axis("off")
    plt.imshow(yen_binary,cmap="gray")
    return (yen_binary,)


@app.cell
def _(plt, yen_binary):
    #remove small from yen
    from skimage.morphology import remove_small_objects
    from skimage.morphology import binary_opening, disk

    opened_holes = binary_opening(yen_binary, disk(10))

    cleaned_yen = remove_small_objects(opened_holes, min_size=900)
    plt.axis("off")
    plt.imshow(cleaned_yen,cmap="gray")
    return


@app.cell
def _(first_image, plt):
    plt.axis("off")
    plt.imshow(first_image)
    return


@app.cell
def _(gray_rescaled, np, plt):
    #multisotsu
    from skimage.filters import threshold_multiotsu

    thresholds = threshold_multiotsu(gray_rescaled, classes=3)
    regions = np.digitize(gray_rescaled, bins=thresholds)

    plt.axis("off")
    plt.imshow(regions,cmap="gray")
    return


if __name__ == "__main__":
    app.run()
