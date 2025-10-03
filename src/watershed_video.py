import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import pandas as pd
import os
from src.watershed_image import watershed_pipe
from rich import print as rprint


input_path = "data/C4.tuns.mkv"
images = iio.imiter(input_path)

output_dir = f"images2/{input_path.split('/')[-1]}"
os.makedirs(output_dir, exist_ok=True)
print(output_dir)

csv_path = f"{output_dir}/objects_data.csv"
csv_columns = ["frame", "label", "centroid_y", "centroid_x", "area"]
with open(csv_path, "w") as f:
    pd.DataFrame(columns=csv_columns).to_csv(f, index=False)

frame_num = 0
for image in images:
    rprint(f"Processing frame {frame_num}")

    try:
        image = image[:, 250:1500, :]

        labels = watershed_pipe(image=image)

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]

        num_objects = len(unique_labels)
        rprint(f"Frame {frame_num}: {num_objects} objects")

        object_data = []
        centroids = []
        for label in unique_labels:
            label_mask = labels == label
            centroid = ndi.center_of_mass(label_mask)
            area = np.sum(label_mask)
            object_data.append(
                {
                    "frame": frame_num,
                    "label": label,
                    "centroid_y": centroid[0],
                    "centroid_x": centroid[1],
                    "area": area,
                }
            )
            centroids.append(centroid)
        centroids = np.array(centroids)

        if object_data:
            pd.DataFrame(object_data).to_csv(
                csv_path, mode="a", header=False, index=False
            )

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for label in unique_labels:
            label_mask = labels == label
            colored_mask = np.zeros((*label_mask.shape, 4))
            color = np.random.rand(3)
            colored_mask[label_mask, :3] = color
            colored_mask[label_mask, 3] = 0.5
            plt.imshow(colored_mask)

        if centroids.size > 0:
            plt.scatter(centroids[:, 1], centroids[:, 0], c="red", s=20, marker="o")

        output_frame_path = os.path.join(output_dir, f"frame_{frame_num:06d}.png")
        plt.savefig(output_frame_path, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        print(f"Error in frame {frame_num}: {e}")

    frame_num += 1

    # if frame_num == 21:
    #     break

print(f"Frames saved to {output_dir}")
print(f"Object data saved to {csv_path}")
