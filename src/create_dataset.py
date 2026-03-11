"""Script to extract and label first frame of video.

Run with uv:
```python
uv run python -m src.create_dataset -i data/Control -o dataset/Control
```

Run with `--help` flag to see all params:
```python
uv run python -m src.create_dataset --help
```
"""

import os
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import imageio.v3 as iio
import numpy as np
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.info import loginfo
from src.watershed.process import watershed_pipe
from src.watershed.process_gpu import watershed_pipe as watershed_gpu


def process_video(args):
    video_path, output_dir, pipe_f = args
    try:
        frame_iter = iio.imiter(video_path)
        first_image = next(iter(frame_iter))

        labels = pipe_f(image=first_image)
        stem = Path(video_path).stem
        image_path = f"{output_dir}/{stem}.png"
        labels_path = f"{output_dir}/{stem}.npy"

        iio.imwrite(image_path, first_image)
        np.save(labels_path, labels)
        loginfo(
            f"{Path(video_path).name}: extracted first frame and saved image + labels"
        )
        return None
    except Exception as e:
        error_msg = f"Error processing {Path(video_path).name}: {str(e)}"
        loginfo(error_msg)
        return error_msg


class Arguments(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)
    output_path: Annotated[
        str,
        Field(
            title="Output directory",
            validation_alias=AliasChoices("o", "output_dir"),
        ),
    ] = "dataset"
    input_path: Annotated[
        Path,
        Field(
            title="Input directory with videos",
            validation_alias=AliasChoices("i", "input_path"),
        ),
    ]
    pattern: Annotated[
        str,
        Field(
            title="Video file pattern",
            validation_alias=AliasChoices("p", "pattern"),
        ),
    ] = "*"
    n_workers: Annotated[
        int,
        Field(
            title="Number of workers",
            validation_alias=AliasChoices("n", "n_workers"),
        ),
    ] = 4
    gpu: Annotated[
        bool,
        Field(
            title="GPU",
            validation_alias=AliasChoices("g", "gpu"),
        ),
    ] = False


def main():
    args = Arguments()  # ty:ignore[missing-argument]
    num_workers = max(1, args.n_workers)
    loginfo(f"Creating dataset from first frame of videos using {num_workers} workers")

    if args.gpu:
        loginfo("Running with GPU.")
        pipe_f = watershed_gpu
    else:
        pipe_f = watershed_pipe

    os.makedirs(args.output_path, exist_ok=True)

    video_files = list(args.input_path.glob(args.pattern))
    loginfo(f"Found {len(video_files)} videos")

    if not video_files:
        loginfo("No videos found")
        return

    batch_args = [(str(f), args.output_path, pipe_f) for f in video_files]
    all_errors = []

    with Pool(num_workers) as pool:
        batch_results = pool.map(process_video, batch_args)
        for err in batch_results:
            if err:
                all_errors.append(err)

    loginfo(f"Processed {len(video_files)} videos, saved to {args.output_path}")

    if all_errors:
        loginfo(f"Encountered {len(all_errors)} errors")
    else:
        loginfo("No errors encountered")


if __name__ == "__main__":
    main()
