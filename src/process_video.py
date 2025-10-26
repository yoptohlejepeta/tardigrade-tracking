from pathlib import Path
from typing import Annotated
import imageio.v3 as iio
import numpy as np
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import pandas as pd
import os
from skimage.measure import regionprops
from src.info import loginfo
from src.save import save_image
from src.watershed.process import watershed_pipe
from multiprocessing import Pool


def process_frame(args):
    """Worker function: Process a single frame and return data."""
    frame_num, image, output_dir = args
    try:
        labels = watershed_pipe(image=image)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]  # exclude background
        num_objects = len(unique_labels)
        loginfo(f"Frame {frame_num}: {num_objects} objects")

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
                    "compactness": (perimeter**2) / area,
                    "solidity": region.solidity,
                    "sphericity": (4 * np.pi * area) / (perimeter**2),
                    "n_objects": len(regions),
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
        loginfo(f"Error in frame {frame_num}: {str(e)}")
        return [], f"Error in frame {frame_num}: {str(e)}"


def save_object_data(object_data, csv_path):
    """Save object data to CSV."""
    try:
        if object_data:
            df = pd.DataFrame(object_data)
            df.to_csv(csv_path, mode="a", header=False, index=False)
            loginfo(f"Saved {len(object_data)} object records to {csv_path}")
        else:
            loginfo("No object data to save in this batch.")
    except Exception as e:
        loginfo(f"Failed to save CSV: {str(e)}")
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
        Path,
        Field(
            title="Input video",
            description="Path to video",
            validation_alias=AliasChoices("i", "input_path"),
        ),
    ]
    pattern: Annotated[
        str,
        Field(
            title="Pattern",
            description="Pattern for input source",
            validation_alias=AliasChoices("p", "pattern"),
        ),
    ] = "*"
    n_workers: Annotated[
        int,
        Field(
            title="Number of workers",
            description="Number for `processes` param of `Pool` class.",
            validation_alias=AliasChoices("n", "n_workers"),
        ),
    ]


def main():
    args = Arguments()  # pyright: ignore[reportCallIssue]

    num_workers = max(1, args.n_workers)
    batch_size = 50
    loginfo(f"Using {num_workers} workers with batch size {batch_size}")

    for file in args.input_path.glob(args.pattern):
        frame_iter = iio.imiter(file)
        batch_args = []
        all_errors = []
        frame_num = 0

        output_dir = f"{args.output_path}/{file.name}"
        os.makedirs(output_dir, exist_ok=True)
        loginfo(f"Output directory: {output_dir}")
        csv_path = f"{output_dir}/{file.stem}.csv"
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
            "minor_axis_length",
            "perimeter",
            "compactness",
            "solidity",
            "sphericity",
            "n_objects",
        ]
        pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False)  # pyright: ignore[reportArgumentType]

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
            loginfo(
                f"Encountered {len(all_errors)} errors. First few: {all_errors[:3]}"
            )
        else:
            loginfo("No errors encountered.")
        loginfo(f"Frames saved to {output_dir}")
        loginfo(f"Object data saved to {csv_path}")


if __name__ == "__main__":
    main()
