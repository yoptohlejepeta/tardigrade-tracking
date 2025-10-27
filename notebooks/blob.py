import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import imageio.v3 as iio
    import matplotlib.pyplot as plt

    image = iio.imread("data/C2.tuns.tif")

    plt.axis("off")
    plt.imshow(image)
    return image, plt


@app.cell
def _(image, plt):
    red, g, b = image[:,:,0],image[:,:,1],image[:,:,2]
    gray = 1/3*red + 1/3*g + 1/3*b

    plt.axis("off")
    plt.imshow(gray, cmap="gray")
    return (gray,)


@app.cell
def _(gray, image, plt):
    from skimage.feature import blob_dog, blob_log, blob_doh
    from math import sqrt


    image_gray = gray

    blobs_log = blob_log(image_gray, max_sigma=50, num_sigma=10, threshold=0.5)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(image_gray, max_sigma=50, threshold=0.5)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image_gray, max_sigma=100, threshold=20)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian', 'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
