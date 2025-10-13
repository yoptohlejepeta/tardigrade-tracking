import marimo

__generated_with = "0.15.5"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import requests

    def download_sam(model_path: str):
        import os
        if not os.path.exists(model_path):
            # url = https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_t.pt
            url = f"https://dl.fbaipublicfiles.com/segment_anything/{model_path}"
            print("Downloading SAM model...")
            response = requests.get(url, stream=True)
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Done")

    model_path = "sam_vit_b_01ec64.pth"
    download_sam(model_path=model_path)
    return


@app.cell
def _():
    import numpy as np
    import torch
    import cv2
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    import imageio.v3 as iio
    import matplotlib.pyplot as plt
    from skimage.filters import threshold_otsu

    # model_type = "vit_b"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # sam = sam_model_registry[model_type](checkpoint=model_path)
    # sam.to(device=device)

    # mask_generator = SamAutomaticMaskGenerator(
    #     model=sam,
    #     points_per_side=32,
    #     pred_iou_thresh=0.86,
    #     stability_score_thresh=0.92,
    #     crop_n_layers=1,
    #     crop_n_points_downscale_factor=2,
    #     min_mask_region_area=1000, 
    # )

    # # Load image
    image = iio.imread("data/C2.tuns.tif")
    # image = image[:, 0:1500, :]

    # if len(image.shape) == 3:
    #     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # else:
    #     gray_image = image

    # otsu_threshold = threshold_otsu(gray_image)

    # masks = mask_generator.generate(image)

    # filtered_masks = []
    # for mask in masks:
    #     area = mask['area']
    #     bbox = mask['bbox']

    #     if 1000 < area < 10000:
    #         width, height = bbox[2], bbox[3]
    #         aspect_ratio = max(width, height) / min(width, height)

    #         if aspect_ratio < 3:
    #             mask_pixels = gray_image[mask['segmentation']]

    #             avg_intensity = np.mean(mask_pixels)

    #             if avg_intensity > otsu_threshold:
    #                 filtered_masks.append(mask)

    # print(f"Found {len(filtered_masks)} potential tardigrades")
    return image, np, plt


@app.cell
def _():
    # centroids = []

    # for mask2 in filtered_masks:
    #     y_indices, x_indices = np.where(mask2['segmentation'])
    #     centroid_x = int(np.mean(x_indices))
    #     centroid_y = int(np.mean(y_indices))
    #     centroids.append((centroid_x, centroid_y))

    # print(centroids)
    return


@app.cell
def _():
    # def show_masks(image, masks):
    #     plt.figure(figsize=(12, 8))
    #     plt.imshow(image)

    #     for mask in masks:
    #         m = mask['segmentation']
    #         color = [0,0,1]

    #         img = np.ones((m.shape[0], m.shape[1], 3))
    #         for i in range(3):
    #             img[:,:,i] = color[i]
    #         plt.imshow(np.dstack((img, m*0.5)))

    #     for centroid in centroids:
    #         plt.plot(centroid[0], centroid[1], 'ro')

    #     plt.axis('off')
    #     plt.show()

    # show_masks(image, filtered_masks)
    return


@app.cell
def _():
    from ultralytics import SAM

    model = SAM("sam2.1_b.pt")

    results = model("data/C1.72Reh.tif")
    return (results,)


@app.cell
def _(image, np, plt, results):
    result = results[0]

    # Extract masks (convert to numpy for compatibility)
    masks = result.masks.data.cpu().numpy()  # Shape: [num_masks, height, width]

    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    for mask in masks:
        # Threshold mask to binary
        binary_mask = (mask > 0.5).astype(np.uint8)
        # Create blue overlay
        overlay = np.zeros_like(image)
        overlay[binary_mask] = [0, 0, 255]  # Blue in RGB
        plt.imshow(overlay, alpha=0.5)  # Overlay with 50% transparency

    plt.axis('off')
    plt.title(f"Detected Objects: {len(masks)}")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
