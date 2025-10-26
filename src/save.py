from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import dilation, disk
from skimage.segmentation import find_boundaries


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

    font = ImageFont.load_default(size=24)

    if len(centroids) > 0:
        for (cy, cx), label in zip(centroids, unique_labels):
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(255, 0, 0, 255))

            label_text = str(label)
            bbox = draw.textbbox((cx, cy), label_text, font=font)
            text_height = bbox[3] - bbox[1]

            text_x = cx + 8
            text_y = cy - text_height // 2

            draw.text((text_x, text_y), label_text, fill=(255, 255, 0, 255), font=font)

    img.save(output_dir, "PNG")


def save_image_with_boundaries(
    image: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    track_ids: np.ndarray,
    output_path: Path,
    boundary_thickness: int = 2,
) -> None:
    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    boundaries = find_boundaries(labels, mode="outer")
    boundaries = dilation(boundaries, disk(boundary_thickness))
    boundary_coords = np.argwhere(boundaries)
    
    for y, x in boundary_coords:
        draw.point((x, y), fill=(255, 0, 0))
    
    font = ImageFont.load_default(size=24)
    
    if len(centroids) > 0:
        for (cy, cx), track_id in zip(centroids, track_ids):
            draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(255, 0, 0))
            
            label_text = str(int(track_id))
            bbox = draw.textbbox((cx, cy), label_text, font=font)
            text_height = bbox[3] - bbox[1]
            
            text_x = cx + 8
            text_y = cy - text_height // 2
            
            draw.text((text_x, text_y), label_text, fill=(0, 255, 0), font=font)
    
    img.save(output_path, "PNG")
