import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import imageio.v3 as iio
    from skimage.segmentation import chan_vese
    from skimage.filters import threshold_multiotsu, threshold_otsu
    return chan_vese, iio, np, plt, threshold_multiotsu, threshold_otsu


@app.cell
def _(plt):
    def plot_image(image, cmap=None):
        plt.axis("off")
        plt.imshow(image, cmap=cmap)
        plt.show()
    return (plot_image,)


@app.cell
def _(iio, plt):
    image = iio.imread("data/T7.24Reh.mkv", index=0)

    plt.axis("off")
    plt.imshow(image)
    return (image,)


@app.cell
def _(image, plot_image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    gray = 1/3 * r + 1/3 * g + 1/3 * b


    plot_image(gray, cmap="gray")
    return (gray,)


@app.cell
def _(gray, np, plot_image, threshold_multiotsu):
    multiotsu = threshold_multiotsu(gray, classes=3)
    regions = np.digitize(gray, bins=multiotsu)
    plot_image(regions, cmap="gray")
    return


@app.cell
def _(gray, plot_image, threshold_otsu):
    otsu_image = gray > threshold_otsu(gray)
    plot_image(otsu_image, cmap="gray")
    return


@app.cell
def _(chan_vese, gray, plot_image):
    chan = chan_vese(gray, mu=0.8, lambda1=0.6, lambda2=1, extended_output=True,)

    plot_image(chan[0], cmap="gray")
    return


if __name__ == "__main__":
    app.run()
