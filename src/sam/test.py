import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from ultralytics import SAM
from ultralytics.models.sam import SAM2VideoPredictor
from scipy.ndimage import center_of_mass

# Set PyTorch config for memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Paths
image_path = "data/C2.72Reh.tif"  # Initial frame for segmentation
video_path = "data/C2.72Reh.mkv"  # Input video path
output_dir = "output_frames"  # Directory to save annotated frames

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Step 1: Segment initial frame and compute centroids
model = SAM("sam2.1_b.pt")
model.to(device="cuda")
results = model(image_path)

# Extract masks
masks = results[0].masks.data.cpu().numpy()  # Shape: [num_masks, height, width]

# Compute centroids
centroids = []
for mask in masks:
    binary_mask = (mask > 0.5).astype(np.uint8)
    cy, cx = center_of_mass(binary_mask)
    centroids.append([cx, cy])  # Store as [x, y]

num_objects = len(centroids)
print(f"Detected {num_objects} objects in initial frame.")

plotted_image = results[0].plot(masks=True, boxes=False)
plt.figure(figsize=(12, 8))
plt.imshow(plotted_image)
plt.axis("off")
plt.title(f"Detected Objects: {num_objects}")
for i, (cx, cy) in enumerate(centroids, 1):
    plt.scatter(cx, cy, color="red", s=20, marker="o")
    plt.text(
        cx + 10,
        cy - 10,
        str(i),
        color="white",
        fontsize=10,
        weight="bold",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
    )
plt.show()

# Clear GPU memory after initial segmentation
del model
torch.cuda.empty_cache()

overrides = dict(task="segment", mode="predict", model="sam2_b.pt")
predictor = SAM2VideoPredictor(overrides=overrides)

points = centroids
labels = [1 for _ in range(num_objects)]  # Unique labels: [1, 2, 3]

with torch.cuda.amp.autocast():
    results = predictor(source=video_path, points=points, labels=labels, stream=True)

colors = [
    (0, 0, 255),  # Blue
    (0, 255, 0),  # Green
    (255, 0, 0),  # Red
]  # Limited colors for max_objects=3
colors = [colors[i % len(colors)] for i in range(num_objects)]  # Cycle through colors

# Step 5: Process results and save each frame as an image
frame_idx = 0
for result in results:
    frame_bgr = cv2.cvtColor(result.orig_img, cv2.COLOR_RGB2BGR)  # Convert frame to BGR
    masks = (
        result.masks.data.cpu().numpy() if result.masks is not None else []
    )  # Shape: [num_objects, height, width]

    # Check if tracker_id is available
    tracker_ids = (
        result.boxes.id.cpu().numpy()
        if hasattr(result.boxes, "id") and result.boxes.id is not None
        else np.arange(len(masks))
    )

    # Draw masks and labels
    for obj_idx, (mask, tracker_id) in enumerate(zip(masks, tracker_ids)):
        color = colors[tracker_id % len(colors)]  # Unique color per object
        mask = (mask > 0.5).astype(np.uint8)  # Binary mask

        # Create colored overlay for mask
        overlay = np.zeros_like(frame_bgr)
        overlay[mask == 1] = color
        frame_bgr = cv2.addWeighted(frame_bgr, 1.0, overlay, 0.5, 0)

        # Compute centroid for label placement
        cy, cx = center_of_mass(mask)
        label = str(tracker_id + 1)  # Start labels from 1
        cv2.putText(
            frame_bgr,
            label,
            (int(cx) + 10, int(cy) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # Save frame as image
    frame_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.png")
    cv2.imwrite(frame_path, frame_bgr)
    frame_idx += 1

print(f"Saved {frame_idx} annotated frames to {output_dir}")
