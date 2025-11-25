import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import cupy as cp
    from cucim.skimage.filters import unsharp_mask
    import matplotlib.pyplot as plt
    import imageio.v3 as iio

    img = iio.imread('./data/C2.tuns.mkv', index=0)

    # plt.imshow(cp.asnumpy(unsharped_gpu))
    return cp, img, plt, unsharp_mask


@app.cell
def _(cp, img, unsharp_mask):
    gpu_img = cp.asarray(img)

    unsharped = unsharp_mask(gpu_img, amount=3, channel_axis=0)
    return gpu_img, unsharped


@app.cell
def _(cp, plt, unsharped):
    plt.axis("off")
    plt.imshow(cp.asnumpy(unsharped))
    return


@app.cell
def _(gpu_img):
    from cucim.skimage.filters import gaussian

    gauss = gaussian(gpu_img, sigma=3, channel_axis=0)
    return (gauss,)


@app.cell
def _(cp, gauss, plt):
    plt.axis("off")
    plt.imshow(cp.asnumpy(gauss))
    return


if __name__ == "__main__":
    app.run()
