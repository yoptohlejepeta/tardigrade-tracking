import marimo

__generated_with = "0.15.5"
app = marimo.App(width="full")


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
    return image, plt


@app.cell
def _(image, plt):
    from skimage.filters import unsharp_mask, gaussian

    unsharped = unsharp_mask(image, amount=2)

    plt.axis("off")
    plt.imshow(unsharped)
    return gaussian, unsharped


@app.cell
def _(gaussian, plt, unsharped):
    unsh_gaus = gaussian(unsharped, sigma=3, preserve_range=True)

    plt.axis("off")
    plt.imshow(unsh_gaus)

    return (unsh_gaus,)


@app.cell
def _(plt, unsh_gaus):
    prep = unsh_gaus

    r, g, b = prep[:,:,0], prep[:,:,1], prep[:,:,2]
    gray = 1/3*r + 1/3*g + 1/3*b

    plt.axis("off")
    plt.imshow(gray,cmap="gray")
    return (gray,)


@app.cell
def _(gray, plt):
    from skimage.filters import threshold_otsu

    binarized = gray

    otsu_image = binarized > threshold_otsu(binarized)

    plt.axis("off")
    plt.imshow(otsu_image)
    return


@app.cell
def _():
    from skimage.morphology import remove_small_objects, binary_closing, disk
    return


if __name__ == "__main__":
    app.run()
