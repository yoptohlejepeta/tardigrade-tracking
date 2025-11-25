import marimo

__generated_with = "0.17.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import imageio.v3 as iio
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage as ndi

    first_image = iio.imread("data/FC2.0.mkv", index=0)
    # first_image = first_image[:, 300:1500]

    plt.axis('off')
    plt.imshow(first_image)
    return first_image, iio, np, plt


@app.cell
def _(first_image, plt):
    r, g, b = first_image[:,:,0], first_image[:,:,1], first_image[:,:,2]
    gray = 1/3*r + 1/3*g + 1/3*b

    # from skimage.color import rgb2gray

    # gray = rgb2gray(first_image)

    plt.axis("off")
    plt.imshow(gray,cmap="gray")
    return (gray,)


@app.cell
def _(gray, np):
    v_min, v_max = np.percentile(gray, (2, 98))

    v_min, v_max
    return


@app.cell
def _(gray, plt):
    plt.figure(figsize=(10, 5))

    # .ravel() flattens the 2D image into a 1D array
    # bins=256 creates one bar for every possible value (0-255)
    plt.hist(gray.ravel(), bins=256, color='gray', alpha=0.7)

    plt.title("Grayscale Intensity Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency (Pixel Count)")
    plt.xlim([0, 255]) # Limit x-axis to standard pixel range
    plt.show()
    return


@app.cell
def _(gray, plt):
    from skimage.exposure import rescale_intensity
    from skimage.io import imsave
    from skimage import img_as_ubyte


    gray_rescaled = rescale_intensity(gray, in_range=(40, 200), out_range=(0, 255))
    plt.axis("off")
    plt.imshow(gray_rescaled,cmap="gray")

    plt.imsave("data/gray_rescaled_fc2.png", gray_rescaled, cmap="gray")
    return gray_rescaled, rescale_intensity


@app.cell
def _():
    # from skimage.color import rgb2gray
    # import os

    # output_dir = "data/frames_processed"
    # os.makedirs(output_dir, exist_ok=True)

    # def process_frame(frame):
    #     frame = frame[:, 300:1500]

    #     gray = 1/3*frame[:,:,0] + 1/3*frame[:,:,1] + 1/3*frame[:,:,2]

    #     gray_rescaled = rescale_intensity(gray, in_range=(65, 200))

    #     return gray_rescaled

    # video_frames = iio.imiter("data/FC1.0.mkv")

    # print(f"Saving frames to {output_dir}...")

    # for i, frame in enumerate(video_frames):
    #     processed = process_frame(frame)

    #     plt.imsave(f"{output_dir}/frame_{i:05d}.png", processed, cmap="gray")

    # print("Done.")
    return


@app.cell
def _(gray_rescaled, plt):
    from skimage.filters import threshold_yen

    yenThresh = threshold_yen(gray_rescaled)
    binaryYen = gray_rescaled > yenThresh

    plt.axis("off")
    plt.imshow(binaryYen,cmap="gray")
    return binaryYen, threshold_yen


@app.cell
def _(binaryYen, plt):
    # morhp clearing
    from skimage.morphology import remove_small_objects, disk, erosion, opening, closing, dilation

    mask_size = 5
    binaryCleaned = binaryYen
    binaryCleaned = opening(binaryCleaned, footprint=disk(mask_size))
    binaryCleaned = remove_small_objects(binaryCleaned, min_size=600)
    plt.axis("off")
    plt.imshow(binaryCleaned,cmap="gray")
    return disk, opening, remove_small_objects


@app.cell
def _(
    disk,
    first_image,
    opening,
    plt,
    remove_small_objects,
    rescale_intensity,
    threshold_yen,
):
    def binarize_image(image):
        cropped = image[:, 300:1500]

        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        gray = 1/3*r + 1/3*g + 1/3*b

        gray_rescaled = rescale_intensity(gray, in_range=(70, 200), out_range=(0, 255))

        yenThresh = threshold_yen(gray_rescaled)
        binaryYen = gray_rescaled > yenThresh

        mask_size = 5
        binaryCleaned = binaryYen
        binaryCleaned = opening(binaryCleaned, footprint=disk(mask_size))
        binaryCleaned = remove_small_objects(binaryCleaned, min_size=600)

        return binaryCleaned

    processed_image = binarize_image(first_image)

    plt.axis("off")
    plt.imshow(processed_image, cmap="gray")
    return


@app.cell
def _(gray_rescaled, plt):
    from skimage.filters import try_all_threshold

    fig, ax = try_all_threshold(gray_rescaled, figsize=(12, 12), verbose=False)
    plt.show()
    return


@app.cell
def _(first_image, iio):
    # save orig image
    iio.imwrite("data/orig_image.png", first_image)
    return


if __name__ == "__main__":
    app.run()
