import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from src.process import watershed_pipe


if __name__ == "__main__":
    image = iio.imread("data/C1.72Reh.tif")
    image = image[:, 250:1500, :]

    labels = watershed_pipe(image)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]  # exclude background
    centroids = []
    for label in unique_labels:
        centroid = ndi.center_of_mass(labels == label)
        centroids.append(centroid)
    centroids = np.array(centroids)

    plt.imshow(image)

    for label in unique_labels:
        label_mask = labels == label
        colored_mask = np.zeros((*label_mask.shape, 4))
        color = np.random.rand(3)
        colored_mask[label_mask, :3] = color
        colored_mask[label_mask, 3] = 0.5
        plt.imshow(colored_mask)

    plt.scatter(centroids[:, 1], centroids[:, 0], c="red", s=20, marker="o")
    plt.show()
