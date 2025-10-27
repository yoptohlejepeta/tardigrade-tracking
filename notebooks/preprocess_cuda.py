import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import imageio.v3 as iio
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage as ndi
    import cupy as cp
    return cp, iio, np, plt


@app.cell
def _(np, plt):
    def plot_image(image: np.ndarray):
        plt.axis("off")
        plt.imshow(image)
        plt.show()
    return (plot_image,)


@app.cell
def _(cp, iio, plot_image):
    # image = iio.imread("data/C4.tuns.tif")
    image = iio.imread("data/T7.24Reh.mkv", index=0)
    image_gpu = cp.asarray(image)
    plot_image(image)
    return (image_gpu,)


@app.cell
def _(image_gpu, plot_image):
    from cucim.skimage.filters import unsharp_mask

    unsharped = unsharp_mask(image_gpu, amount=3, channel_axis=-1)

    plot_image(unsharped)
    return


if __name__ == "__main__":
    app.run()
