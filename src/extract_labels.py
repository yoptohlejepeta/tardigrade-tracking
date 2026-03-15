import os
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import imageio.v3 as iio
import numpy as np
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.log import loginfo
from src.watershed.process import watershed_pipe
from src.watershed.process_gpu import watershed_pipe as watershed_gpu


def process_frame(args):
    frame_num, image, output_dir, pipe_f = args
    try:
        labels = pipe_f(image=image)
        output_path = f"{output_dir}/frame_{frame_num:06d}.npy"
        np.save(output_path, labels)
        loginfo(f"Frame {frame_num}: extracted and saved labels")
        return None
    except Exception as e:
        loginfo(f"Error in frame {frame_num}: {str(e)}")
        return str(e)


class Arguments(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)
    output_path: Annotated[
        str,
        Field(
            title="Output directory",
            validation_alias=AliasChoices("o", "output_dir"),
        ),
    ] = "labels"
    input_path: Annotated[
        Path,
        Field(
            title="Input video",
            validation_alias=AliasChoices("i", "input_path"),
        ),
    ]
    pattern: Annotated[
        str,
        Field(
            title="Pattern",
            validation_alias=AliasChoices("p", "pattern"),
        ),
    ] = "*"
    n_workers: Annotated[
        int,
        Field(
            title="Number of workers",
            validation_alias=AliasChoices("n", "n_workers"),
        ),
    ]
    frame_step: Annotated[
        int,
        Field(
            title="Process every nth frame",
            validation_alias=AliasChoices("s", "frame_step"),
        ),
    ] = 1
    gpu: Annotated[
        bool,
        Field(
            title="GPU",
            validation_alias=AliasChoices("g", "gpu"),
        ),
    ] = False
    max_seconds: Annotated[
        int,
        Field(
            title="Max seconds to extract (from start)",
            validation_alias=AliasChoices("t", "max_seconds"),
        ),
    ] = 60


def main():
    args = Arguments()  # ty:ignore[missing-argument]
    num_workers = max(1, args.n_workers)
    batch_size = 50
    loginfo(f"Using {num_workers} workers with batch size {batch_size}")

    if args.gpu:
        loginfo("Running with GPU.")
        pipe_f = watershed_gpu
    else:
        pipe_f = watershed_pipe

    for file in args.input_path.glob(args.pattern):
        # Get video FPS to calculate max frames
        meta = iio.immeta(file)
        fps = meta.get("fps", 30)  # default to 30 if not available
        max_frames = int(fps * args.max_seconds)
        loginfo(
            f"Video FPS: {fps}, processing first {args.max_seconds}s ({max_frames} frames)"
        )

        frame_iter = iio.imiter(file)
        batch_args = []
        all_errors = []
        frame_num = 0

        output_dir = f"{args.output_path}/{file.stem}"
        os.makedirs(output_dir, exist_ok=True)
        loginfo(f"Output directory: {output_dir}")

        with Pool(num_workers) as pool:
            for image in frame_iter:
                if frame_num >= max_frames:
                    break
                if frame_num % args.frame_step == 0:
                    batch_args.append((frame_num, image, output_dir, pipe_f))
                    if len(batch_args) >= batch_size:
                        batch_results = pool.map_async(process_frame, batch_args).get()
                        for err in batch_results:
                            if err:
                                all_errors.append(err)
                        batch_args = []
                frame_num += 1

            if batch_args:
                batch_results = pool.map(process_frame, batch_args)
                for err in batch_results:
                    if err:
                        all_errors.append(err)

        loginfo(f"Processed {frame_num} frames, saved to {output_dir}")

        if all_errors:
            loginfo(f"Encountered {len(all_errors)} errors")
        else:
            loginfo("No errors encountered")


if __name__ == "__main__":
    main()
