import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu, unsharp_mask
from skimage.morphology import disk, opening, remove_small_objects
from skimage.segmentation import clear_border, watershed


def watershed_pipe(image: np.ndarray) -> np.ndarray:
    """Process pipeline for image watershed segmentation.

    image -> grayscale -> otsu -> dilation -> remove_small -> distance + watershed

    """
    height, width = image.shape[:2]
    image[int(height * 0.8) :, int(width * 0.7) :] = 0

    unsharped = unsharp_mask(image, amount=3)
    r, g, b = unsharped[:, :, 0], unsharped[:, :, 1], unsharped[:, :, 2]

    gray = (1 / 3 * r) + (1 / 3 * g) + (1 / 3 * b)

    gauss = gaussian(gray, sigma=3)

    otsu_image_gaus = gauss > threshold_otsu(gauss)

    removed = opening(otsu_image_gaus, footprint=disk(10))
    removed = remove_small_objects(removed, min_size=2000)

    distance = ndi.distance_transform_edt(removed)
    coords = peak_local_max(
        distance,
        labels=removed,
        min_distance=20,
        footprint=disk(40),
        exclude_border=True,
    )
    mask = np.zeros(distance.shape, dtype=bool)  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(input=mask)  # pyright: ignore[reportGeneralTypeIssues]
    labels = watershed(-distance, markers, mask=removed, connectivity=2)  # pyright: ignore
    labels = clear_border(labels)

    return labels
