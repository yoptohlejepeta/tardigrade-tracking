from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def save_image(
    image: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    output_dir: Path,
) -> None:
    img = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    for label in unique_labels:
        label_mask = (labels == label).astype(np.uint8) * 255
        mask_img = Image.fromarray(label_mask).convert("L")
        blue_layer = Image.new("RGBA", img.size, (0, 0, 255, 128))
        overlay.paste(blue_layer, (0, 0), mask=mask_img)

    img = Image.alpha_composite(img, overlay)

    draw = ImageDraw.Draw(img)
    if len(centroids) > 0:
        for cy, cx in centroids:
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(255, 0, 0, 255))

    img.save(output_dir, "PNG")
