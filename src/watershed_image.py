import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import dilation
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

image = iio.imread("data/C2.tuns.tif")
image = image[:, 250:1500, :]


def process(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    img_part = 1 / 3 * r + 1 / 3 * g + 1 / 3 * b
    otsu_image = img_part > threshold_otsu(img_part)

    closed = otsu_image
    for _ in range(5):
        closed = dilation(otsu_image, footprint=np.ones((5, 5)))

    removed = remove_small_objects(closed, min_size=2500)

    final_image = removed

    distance = ndi.distance_transform_edt(final_image)
    coords = peak_local_max(distance, labels=final_image, min_distance=30)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=final_image, connectivity=2)

    return labels

if __name__ == "__main__":
    labels = process(image)
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
    # plt.show()
