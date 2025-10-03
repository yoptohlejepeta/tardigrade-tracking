from pathlib import Path
from typing import Annotated
import imageio.v3 as iio
import numpy as np
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import pandas as pd
import os
from skimage.measure import regionprops
from src.save import save_image
from src.watershed_image import watershed_pipe
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from loguru import logger

def process_frame(args):
    """Worker function: Process a single frame and return data."""
    frame_num, image, output_dir = args
    try:
        image = image[:, 0:1500, :]
        labels = watershed_pipe(image=image)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]  # exclude background
        num_objects = len(unique_labels)
        logger.info(f"Frame {frame_num}: {num_objects} objects")

        object_data = []
        centroids = []
        regions = regionprops(labels)
        for region in regions:
            if region.label == 0:
                continue
            area = region.area
            perimeter = region.perimeter
            object_data.append(
                {
                    "frame": frame_num,
                    "label": region.label,
                    "centroid_y": region.centroid[0],
                    "centroid_x": region.centroid[1],
                    "area": area,
                    "eccentricity": region.eccentricity,
                    "extent": region.extent,
                    "major_axis_length": region.major_axis_length,
                    "max_feret_diameter": region.feret_diameter_max,
                    "minor_axis_length": region.minor_axis_length,
                    "perimeter": perimeter,
                    "perimeter_complexity": (perimeter**2) / area,
                    "solidity": region.solidity,
                    "sphericity": (4 * np.pi * area) / (perimeter**2),
                }
            )
            centroids.append(region.centroid)
        centroids = np.array(centroids)

        output_frame_path = Path(output_dir) / f"frame_{frame_num:06d}.png"
        save_image(
            image=image,
            labels=labels,
            centroids=centroids,
            output_dir=output_frame_path,
        )

        return object_data, None
    except Exception as e:
        logger.error(f"Error in frame {frame_num}: {str(e)}")
        return [], f"Error in frame {frame_num}: {str(e)}"


def save_object_data(object_data, csv_path):
    """Save object data to CSV."""
    try:
        if object_data:
            df = pd.DataFrame(object_data)
            df.to_csv(csv_path, mode="a", header=False, index=False)
            logger.info(f"Saved {len(object_data)} object records to {csv_path}")
        else:
            logger.warning("No object data to save in this batch.")
    except Exception as e:
        logger.error(f"Failed to save CSV: {str(e)}")
        raise


class Arguments(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)
    output_path: Annotated[
        str,
        Field(
            title="Output directory",
            validation_alias=AliasChoices("o", "output_dir"),
        ),
    ] = "images"
    input_path: Annotated[
        str,
        Field(
            title="Input video",
            description="Path to video",
            validation_alias=AliasChoices("i", "input_path"),
        ),
    ]


def main():
    args = Arguments()  # pyright: ignore[reportCallIssue]
    output_dir = f"{args.output_path}/{args.input_path.split('/')[-1]}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    csv_path = f"{output_dir}/objects_data.csv"
    csv_columns = [
        "frame",
        "label",
        "centroid_y",
        "centroid_x",
        "area",
        "eccentricity",
        "extent",
        "major_axis_length",
        "max_feret_diameter",
        "min_feret_diameter",
        "minor_axis_length",
        "perimeter",
        "perimeter_complexity",
        "solidity",
        "sphericity",
    ]
    pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)  # pyright: ignore[reportArgumentType]

    num_workers = max(1, cpu_count() - 1)
    batch_size = 50
    logger.info(f"Using {num_workers} workers with batch size {batch_size}")

    try:
        metadata = iio.immeta(args.input_path)
        total_frames = metadata.get("nframes", None)
        logger.info(f"Total frames: {total_frames}")
    except Exception as e:
        logger.warning(
            f"Could not determine total frames: {str(e)}. Progress bar may be inaccurate."
        )
        total_frames = None

    frame_iter = iio.imiter(args.input_path)
    batch_args = []
    all_errors = []
    frame_num = 0

    with Pool(num_workers) as pool:
        for image in frame_iter:
            batch_args.append((frame_num, image, output_dir))
            if len(batch_args) >= batch_size:
                batch_results = pool.map(process_frame, batch_args)
                for batch_data, batch_error in batch_results:
                    save_object_data(batch_data, csv_path)
                    if batch_error:
                        all_errors.append(batch_error)
                batch_args = []
            frame_num += 1

        if batch_args:
            batch_results = pool.map(process_frame, batch_args)
            for batch_data, batch_error in batch_results:
                save_object_data(batch_data, csv_path)
                if batch_error:
                    all_errors.append(batch_error)

    if all_errors:
        logger.warning(
            f"Encountered {len(all_errors)} errors. First few: {all_errors[:3]}"
        )
    else:
        logger.info("No errors encountered.")
    logger.info(f"Frames saved to {output_dir}")
    logger.info(f"Object data saved to {csv_path}")


if __name__ == "__main__":
    main()
