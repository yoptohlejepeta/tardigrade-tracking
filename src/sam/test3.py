import numpy as np
import os
import torch
import imageio.v3 as iio
from ultralytics.models import SAM
import matplotlib.pyplot as plt

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

print("Loading image...")
image = iio.imread("data/C1.72Reh.tif")
print(f"Image shape: {image.shape}")

print("Loading SAM model...")
model = SAM("sam2.1_b.pt")
model.to(device="cuda")

print("Running inference...")
results = model(image)
result = results[0]

masks = result.masks.data.cpu().numpy()
print(f"Number of masks detected: {len(masks)}")

plt.figure(figsize=(12, 8))
plt.imshow(image)

print("Creating overlays...")
for i, mask in enumerate(masks):
    binary_mask = (mask > 0.5).astype(np.uint8)
    overlay = np.zeros_like(image)
    overlay[binary_mask] = [0, 0, 255]
    plt.imshow(overlay, alpha=0.5)

plt.axis("off")
plt.title(f"Detected Objects: {len(masks)}")

output_path = "output_segmentation.png"
print(f"Saving image to {output_path}...")
plt.savefig(output_path)
print("Done!")
