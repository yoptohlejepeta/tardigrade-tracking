import os
from pathlib import Path
from typing import Annotated, Dict, List

import imageio.v3 as iio
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu, unsharp_mask
from skimage.measure import regionprops
from skimage.morphology import disk, opening, remove_small_objects
from skimage.segmentation import clear_border, watershed

matplotlib.use("Agg")


def watershed_segmentation(image: np.ndarray) -> np.ndarray:
    """
    Segment tardigrades using watershed algorithm.
    Reimplementation of the watershed_pipe function.
    """
    height, width = image.shape[:2]
    image[int(height * 0.8) :, int(width * 0.7) :] = 0

    unsharped = unsharp_mask(image, amount=3)

    if len(unsharped.shape) == 3:
        r, g, b = unsharped[:, :, 0], unsharped[:, :, 1], unsharped[:, :, 2]
        gray = (r + g + b) / 3.0
    else:
        gray = unsharped

    gauss = gaussian(gray, sigma=3)
    binary = gauss > threshold_otsu(gauss)

    cleaned = opening(binary, footprint=disk(10))
    cleaned = remove_small_objects(cleaned, min_size=2000)

    distance = ndimage.distance_transform_edt(cleaned)

    coords = peak_local_max(
        distance,
        labels=cleaned,
        min_distance=30,
        footprint=disk(45),
        exclude_border=True,
    )

    mask = np.zeros(distance.shape, dtype=bool)  # pyright: ignore
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)  # pyright: ignore

    labels = watershed(-distance, markers, mask=cleaned, connectivity=1)  # pyright: ignore
    labels = clear_border(labels)

    return labels


class TardigradeTracker:
    def __init__(self, max_distance_factor: float = 0.5):
        self.max_distance_factor = max_distance_factor
        self.next_track_id = 1
        self.previous_objects = {}

    def update(self, regions: List) -> Dict[int, int]:
        if not regions:
            self.previous_objects = {}
            return {}

        current_data = []
        for region in regions:
            if region.label != 0:
                current_data.append(
                    {
                        "label": region.label,
                        "centroid": region.centroid,
                        "area": region.area,
                    }
                )

        if not current_data:
            self.previous_objects = {}
            return {}

        label_to_track = {}

        if not self.previous_objects:
            for item in current_data:
                track_id = self.next_track_id
                label_to_track[item["label"]] = track_id
                self.next_track_id += 1
        else:
            current_centroids = np.array([d["centroid"] for d in current_data])
            current_areas = np.array([d["area"] for d in current_data])
            current_labels = [d["label"] for d in current_data]

            prev_ids = list(self.previous_objects.keys())
            prev_centroids = np.array(
                [self.previous_objects[tid]["centroid"] for tid in prev_ids]
            )
            prev_areas = np.array(
                [self.previous_objects[tid]["area"] for tid in prev_ids]
            )

            distances = cdist(current_centroids, prev_centroids)

            area_matrix = np.sqrt(np.outer(current_areas, prev_areas))
            max_distances = self.max_distance_factor * np.sqrt(area_matrix / np.pi) * 2

            valid_matches = distances <= max_distances

            assigned_current = set()
            assigned_previous = set()

            matches = []
            for i in range(len(current_data)):
                for j in range(len(prev_ids)):
                    if valid_matches[i, j]:
                        matches.append((distances[i, j], i, j))

            matches.sort()

            for dist, curr_idx, prev_idx in matches:
                if (
                    curr_idx not in assigned_current
                    and prev_idx not in assigned_previous
                ):
                    label_to_track[current_labels[curr_idx]] = prev_ids[prev_idx]
                    assigned_current.add(curr_idx)
                    assigned_previous.add(prev_idx)

            for i, label in enumerate(current_labels):
                if i not in assigned_current:
                    label_to_track[label] = self.next_track_id
                    self.next_track_id += 1

        self.previous_objects = {}
        for item in current_data:
            track_id = label_to_track[item["label"]]
            self.previous_objects[track_id] = {
                "centroid": item["centroid"],
                "area": item["area"],
            }

        return label_to_track


def save_tracked_frame(
    image: np.ndarray,
    labels: np.ndarray,
    label_to_track: Dict[int, int],
    regions: List,
    output_path: Path,
) -> None:
    """save image using pillow"""
    try:
        img = Image.fromarray(image).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            font = ImageFont.load_default(size=24)
        except Exception:
            font = ImageFont.load_default()

        track_image = np.zeros_like(labels)
        for seg_label, track_id in label_to_track.items():
            track_image[labels == seg_label] = track_id

        unique_track_ids = np.unique(track_image)
        unique_track_ids = unique_track_ids[unique_track_ids != 0]
        for track_id in unique_track_ids:
            track_mask = (track_image == track_id).astype(np.uint8) * 255
            mask_img = Image.fromarray(track_mask).convert("L")
            blue_layer = Image.new("RGBA", img.size, (0, 0, 255, 128))
            overlay.paste(blue_layer, (0, 0), mask=mask_img)

        img = Image.alpha_composite(img, overlay)

        draw = ImageDraw.Draw(img)
        for region in regions:
            if region.label != 0 and region.label in label_to_track:
                track_id = label_to_track[region.label]
                cy, cx = region.centroid
                draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(255, 0, 0, 255))
                label_text = str(track_id)
                bbox = draw.textbbox((cx, cy), label_text, font=font)
                text_height = bbox[3] - bbox[1]
                text_x = cx + 8
                text_y = cy - text_height // 2
                draw.text(
                    (text_x, text_y), label_text, fill=(255, 255, 0, 255), font=font
                )

        img.save(output_path, "PNG")
        print(f"Saved image to {output_path}")
    except Exception as e:
        print(f"Failed to save image {output_path}: {e}")


class Arguments(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)
    output_path: Annotated[
        str,
        Field(
            title="Output directory", validation_alias=AliasChoices("o", "output_dir")
        ),
    ] = "tracked_output"
    input_path: Annotated[
        Path,
        Field(
            title="Input video path", validation_alias=AliasChoices("i", "input_path")
        ),
    ]
    pattern: Annotated[
        str, Field(title="File pattern", validation_alias=AliasChoices("p", "pattern"))
    ] = "*.mp4"
    max_distance_factor: Annotated[
        float,
        Field(
            title="Max distance factor",
            validation_alias=AliasChoices("d", "max_distance"),
        ),
    ] = 0.3
    save_every_n: Annotated[
        int,
        Field(
            title="Save image every N frames",
            validation_alias=AliasChoices("s", "save_every"),
        ),
    ] = 1


def main():
    args = Arguments() # pyright: ignore

    print("Tardigrade Tracking - Standalone Version")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Max movement: {args.max_distance_factor * 100}% of object diameter")
    print(f"Save every {args.save_every_n} frames")
    print("-" * 50)

    for video_file in args.input_path.glob(args.pattern):
        print(f"\nProcessing: {video_file.name}")

        output_dir = Path(args.output_path) / video_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"No write permission for {output_dir}")

        csv_path = output_dir / f"{video_file.stem}_tracking.csv"
        csv_columns = [
            "frame",
            "track_id",
            "centroid_x",
            "centroid_y",
            "area",
            "eccentricity",
            "solidity",
            "perimeter",
        ]

        tracker = TardigradeTracker(max_distance_factor=args.max_distance_factor)

        all_data = []
        frame_num = 0

        try:
            for frame in iio.imiter(video_file):
                print(
                    f"\rProcessing frame {frame_num}, shape={frame.shape}",
                    end="",
                    flush=True,
                )

                labels = watershed_segmentation(frame)
                regions = regionprops(labels)
                print(f", regions={len(regions)}", flush=True)

                label_to_track = tracker.update(regions)

                if frame_num % args.save_every_n == 0:
                    output_path = output_dir / f"tracked_frame_{frame_num:06d}.png"
                    save_tracked_frame(
                        frame, labels, label_to_track, regions, output_path
                    )

                for region in regions:
                    if region.label != 0 and region.label in label_to_track:
                        track_id = label_to_track[region.label]
                        all_data.append(
                            {
                                "frame": frame_num,
                                "track_id": track_id,
                                "centroid_x": region.centroid[1],
                                "centroid_y": region.centroid[0],
                                "area": region.area,
                                "eccentricity": region.eccentricity,
                                "solidity": region.solidity,
                                "perimeter": region.perimeter,
                            }
                        )

                frame_num += 1
        except Exception as e:
            print(f"\nError processing video {video_file}: {e}")
            continue

        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(csv_path, index=False)
            print(f"\nSaved {len(all_data)} tracked objects to {csv_path}")

            track_counts = df.groupby("track_id").size()
            print(f"Total tracks: {len(track_counts)}")
            print(f"Average track length: {track_counts.mean():.1f} frames")
            print(f"Longest track: {track_counts.max()} frames")
        else:
            print(f"\nNo objects tracked in {video_file}")


if __name__ == "__main__":
    main()
