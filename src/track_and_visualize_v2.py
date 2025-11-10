import os
from collections import deque
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects

from src.info import loginfo
from src.save import save_image_with_boundaries


class Arguments(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)
    input_path: Annotated[
        Path,
        Field(
            title="Input directory with .npy files",
            validation_alias=AliasChoices("i", "input_path"),
        ),
    ]
    output_path: Annotated[
        str,
        Field(
            title="Output directory",
            validation_alias=AliasChoices("o", "output_dir"),
        ),
    ] = "images"
    max_distance_percent: Annotated[
        float,
        Field(
            title="Maximum distance as percentage of major axis length",
            validation_alias=AliasChoices("d", "max_distance_percent"),
        ),
    ] = 20.0
    lookback_frames: Annotated[
        int,
        Field(
            title="Number of frames to look back for tracking",
            validation_alias=AliasChoices("l", "lookback"),
        ),
    ] = 5
    n_workers: Annotated[
        int,
        Field(
            title="Number of workers for parallel image saving",
            validation_alias=AliasChoices("n", "n_workers"),
        ),
    ] = 4


def track_objects_with_lookback(
    history, curr_centroids, curr_major_axes, curr_labels, next_id, max_distance_percent
):
    """Optimized tracking function that works with pre-extracted arrays."""
    if len(curr_centroids) == 0:
        return {}, next_id

    id_map = {}
    matched_curr = set()

    for frame_data in history:
        if len(frame_data["centroids"]) == 0:
            continue

        prev_centroids = frame_data["centroids"]
        prev_labels = frame_data["labels"]
        prev_id_map = frame_data["id_map"]

        unmatched_curr_idx = [
            i for i in range(len(curr_centroids)) if i not in matched_curr
        ]
        if not unmatched_curr_idx:
            break

        unmatched_curr_centroids = curr_centroids[unmatched_curr_idx]
        unmatched_curr_major_axes = curr_major_axes[unmatched_curr_idx]
        cost_matrix = cdist(unmatched_curr_centroids, prev_centroids)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for curr_rel_idx, prev_idx in zip(row_ind, col_ind):
            curr_idx = unmatched_curr_idx[curr_rel_idx]
            adaptive_max_distance = unmatched_curr_major_axes[curr_rel_idx] * (
                max_distance_percent / 100.0
            )

            if cost_matrix[curr_rel_idx, prev_idx] < adaptive_max_distance:
                prev_label = prev_labels[prev_idx]
                prev_id = prev_id_map[prev_label]
                curr_label = curr_labels[curr_idx]

                if curr_label not in id_map:
                    id_map[curr_label] = prev_id
                    matched_curr.add(curr_idx)

    # Assign new IDs to unmatched objects
    for curr_idx in range(len(curr_centroids)):
        if curr_idx not in matched_curr:
            curr_label = curr_labels[curr_idx]
            if curr_label not in id_map:
                id_map[curr_label] = next_id
                next_id += 1

    return id_map, next_id


def save_image_worker(args):
    """Worker function for parallel image saving."""
    image, tracked_labels, centroids, track_ids, output_frame_path = args
    try:
        save_image_with_boundaries(
            image=image,
            labels=tracked_labels,
            centroids=centroids,
            track_ids=track_ids,
            output_path=output_frame_path,
        )
        return None
    except Exception as e:
        return f"Error saving frame {output_frame_path}: {str(e)}"


def main():
    args = Arguments()

    npy_files = sorted(args.input_path.glob("frame_*.npy"))
    loginfo(f"Found {len(npy_files)} .npy files")
    loginfo(f"Using lookback of {args.lookback_frames} frames")
    loginfo(f"Max distance: {args.max_distance_percent}% of major axis length")
    loginfo(f"Using {args.n_workers} workers for image saving")

    video_name = args.input_path.name
    output_dir = f"{args.output_path}/{video_name}"
    os.makedirs(output_dir, exist_ok=True)
    loginfo(f"Output directory: {output_dir}")

    csv_path = f"{output_dir}/{video_name}.csv"

    # Optimized: Minimal history storage - only keep what we need
    history = deque(maxlen=args.lookback_frames)
    next_id = 1
    all_data = []

    # Queue for parallel image saving
    save_queue = []

    for frame_num, npy_file in enumerate(npy_files):
        data = np.load(npy_file, allow_pickle=True).item()
        image = data["image"]
        labels = data["labels"]

        labels = remove_small_objects(labels, min_size=2000)
        regions = regionprops(labels)
        regions = [r for r in regions if r.label != 0]

        # Extract arrays once upfront
        if regions:
            curr_centroids = np.array([r.centroid for r in regions])
            curr_major_axes = np.array([r.major_axis_length for r in regions])
            curr_labels = np.array([r.label for r in regions])
        else:
            curr_centroids = np.array([]).reshape(0, 2)
            curr_major_axes = np.array([])
            curr_labels = np.array([])

        id_map, next_id = track_objects_with_lookback(
            history,
            curr_centroids,
            curr_major_axes,
            curr_labels,
            next_id,
            args.max_distance_percent,
        )

        # Optimized: Store only minimal data in history
        history.append(
            {
                "centroids": curr_centroids,
                "major_axes": curr_major_axes,
                "labels": curr_labels,
                "id_map": id_map,
            }
        )

        tracked_labels = np.zeros_like(labels)
        centroids = []
        track_ids = []

        used_track_ids = set()
        for idx, region in enumerate(regions):
            curr_label = curr_labels[idx]
            track_id = id_map.get(curr_label)
            if track_id is None:
                loginfo(
                    f"Warning: region {curr_label} has no track_id mapping in frame {frame_num}"
                )
                continue

            if track_id in used_track_ids:
                loginfo(
                    f"Warning: duplicate track_id {track_id} in frame {frame_num}, assigning new ID"
                )
                track_id = next_id
                next_id += 1
                id_map[curr_label] = track_id

            used_track_ids.add(track_id)
            mask = labels == region.label
            tracked_labels[mask] = track_id
            centroids.append(region.centroid)
            track_ids.append(track_id)

            area = region.area
            perimeter = region.perimeter
            all_data.append(
                {
                    "frame": frame_num,
                    "track_id": track_id,
                    "centroid_y": region.centroid[0],
                    "centroid_x": region.centroid[1],
                    "area": area,
                    "eccentricity": region.eccentricity,
                    "extent": region.extent,
                    "major_axis_length": region.major_axis_length,
                    "max_feret_diameter": region.feret_diameter_max,
                    "minor_axis_length": region.minor_axis_length,
                    "perimeter": perimeter,
                    "compactness": (perimeter**2) / area if area > 0 else 0,
                    "solidity": region.solidity,
                    "sphericity": (4 * np.pi * area) / (perimeter**2)
                    if perimeter > 0
                    else 0,
                    "n_objects": len(regions),
                }
            )

        centroids = np.array(centroids) if centroids else np.array([]).reshape(0, 2)
        track_ids = np.array(track_ids) if track_ids else np.array([])

        output_frame_path = Path(output_dir) / f"frame_{frame_num:06d}.png"

        # Queue image for parallel saving
        save_queue.append(
            (
                image.copy(),  # Copy to avoid memory issues
                tracked_labels.copy(),
                centroids.copy(),
                track_ids.copy(),
                output_frame_path,
            )
        )

        # Process save queue in batches
        if len(save_queue) >= 50 or frame_num == len(npy_files) - 1:
            with Pool(args.n_workers) as pool:
                errors = pool.map(save_image_worker, save_queue)
                errors = [e for e in errors if e is not None]
                if errors:
                    for error in errors[:3]:  # Log first 3 errors
                        loginfo(error)
            save_queue = []

        if frame_num % 100 == 0:
            loginfo(f"Processed {frame_num + 1}/{len(npy_files)} frames")

    # Save all data at once (much faster than incremental writes)
    loginfo("Saving CSV data...")
    df = pd.DataFrame(all_data)
    df.to_csv(csv_path, index=False)
    loginfo(f"Object data saved to {csv_path}")
    loginfo(f"Frames saved to {output_dir}")


if __name__ == "__main__":
    main()
