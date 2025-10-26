from pathlib import Path
from typing import Annotated
import numpy as np
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import pandas as pd
import os
from collections import deque
from skimage.measure import regionprops
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
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


def track_objects_with_lookback(history, curr_regions: list, next_id, max_distance_percent):
    if not curr_regions:
        return {}, next_id
    
    curr_centroids = np.array([r.centroid for r in curr_regions])
    curr_major_axes = np.array([r.major_axis_length for r in curr_regions])
    id_map = {}
    matched_curr = set()
    
    for frame_data in history:
        if not frame_data['regions']:
            continue
            
        prev_centroids = np.array([r.centroid for r in frame_data['regions']])
        prev_id_map = frame_data['id_map']
        
        unmatched_curr_idx = [i for i in range(len(curr_regions)) if i not in matched_curr]
        if not unmatched_curr_idx:
            break
        
        unmatched_curr_centroids = curr_centroids[unmatched_curr_idx]
        unmatched_curr_major_axes = curr_major_axes[unmatched_curr_idx]
        cost_matrix = cdist(unmatched_curr_centroids, prev_centroids)
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for curr_rel_idx, prev_idx in zip(row_ind, col_ind):
            curr_idx = unmatched_curr_idx[curr_rel_idx]
            adaptive_max_distance = unmatched_curr_major_axes[curr_rel_idx] * (max_distance_percent / 100.0)
            
            if cost_matrix[curr_rel_idx, prev_idx] < adaptive_max_distance:
                prev_label = frame_data['regions'][prev_idx].label
                prev_id = prev_id_map[prev_label]
                curr_label = curr_regions[curr_idx].label
                
                if curr_label not in id_map:
                    id_map[curr_label] = prev_id
                    matched_curr.add(curr_idx)
    
    for curr_idx, region in enumerate(curr_regions):
        if curr_idx not in matched_curr:
            if region.label not in id_map:
                id_map[region.label] = next_id
                next_id += 1
    
    return id_map, next_id


def main():
    args = Arguments()
    
    npy_files = sorted(args.input_path.glob("frame_*.npy"))
    loginfo(f"Found {len(npy_files)} .npy files")
    loginfo(f"Using lookback of {args.lookback_frames} frames")
    loginfo(f"Max distance: {args.max_distance_percent}% of major axis length")
    
    video_name = args.input_path.name
    output_dir = f"{args.output_path}/{video_name}"
    os.makedirs(output_dir, exist_ok=True)
    loginfo(f"Output directory: {output_dir}")
    
    csv_path = f"{output_dir}/{video_name}.csv"
    csv_columns = [
        "frame",
        "track_id",
        "centroid_y",
        "centroid_x",
        "area",
        "eccentricity",
        "extent",
        "major_axis_length",
        "max_feret_diameter",
        "minor_axis_length",
        "perimeter",
        "compactness",
        "solidity",
        "sphericity",
        "n_objects",
    ]
    pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)
    
    history = deque(maxlen=args.lookback_frames)
    next_id = 1
    all_data = []
    
    for frame_num, npy_file in enumerate(npy_files):
        data = np.load(npy_file, allow_pickle=True).item()
        image = data["image"]
        labels = data["labels"]
        
        regions = regionprops(labels)
        regions = [r for r in regions if r.label != 0]
        
        id_map, next_id = track_objects_with_lookback(
            history, regions, next_id, args.max_distance_percent
        )
        
        history.append({
            'regions': regions,
            'id_map': id_map,
            'frame_num': frame_num
        })
        
        tracked_labels = np.zeros_like(labels)
        centroids = []
        track_ids = []
        
        used_track_ids = set()
        for region in regions:
            track_id = id_map.get(region.label)
            if track_id is None:
                loginfo(f"Warning: region {region.label} has no track_id mapping in frame {frame_num}")
                continue
            
            if track_id in used_track_ids:
                loginfo(f"Warning: duplicate track_id {track_id} in frame {frame_num}, assigning new ID")
                track_id = next_id
                next_id += 1
                id_map[region.label] = track_id
            
            used_track_ids.add(track_id)
            mask = labels == region.label
            tracked_labels[mask] = track_id
            centroids.append(region.centroid)
            track_ids.append(track_id)
            
            area = region.area
            perimeter = region.perimeter
            all_data.append({
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
                "sphericity": (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0,
                "n_objects": len(regions),
            })
        
        centroids = np.array(centroids) if centroids else np.array([]).reshape(0, 2)
        track_ids = np.array(track_ids) if track_ids else np.array([])
        
        output_frame_path = Path(output_dir) / f"frame_{frame_num:06d}.png"
        save_image_with_boundaries(
            image=image,
            labels=tracked_labels,
            centroids=centroids,
            track_ids=track_ids,
            output_path=output_frame_path,
        )
        
        loginfo(f"Frame {frame_num}: {len(regions)} objects tracked")
    
    df = pd.DataFrame(all_data)
    df.to_csv(csv_path, index=False)
    loginfo(f"Object data saved to {csv_path}")
    loginfo(f"Frames saved to {output_dir}")


if __name__ == "__main__":
    main()
