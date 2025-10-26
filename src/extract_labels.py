from pathlib import Path
from typing import Annotated
import imageio.v3 as iio
import numpy as np
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from src.info import loginfo
from src.watershed.process import watershed_pipe
from multiprocessing import Pool


def process_frame(args):
    frame_num, image, output_dir = args
    try:
        labels = watershed_pipe(image=image)
        output_path = f"{output_dir}/frame_{frame_num:06d}.npy"
        np.save(output_path, {"image": image, "labels": labels})
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


def main():
    args = Arguments()
    num_workers = max(1, args.n_workers)
    batch_size = 50
    loginfo(f"Using {num_workers} workers with batch size {batch_size}")

    for file in args.input_path.glob(args.pattern):
        frame_iter = iio.imiter(file)
        batch_args = []
        all_errors = []
        frame_num = 0

        output_dir = f"{args.output_path}/{file.stem}"
        os.makedirs(output_dir, exist_ok=True)
        loginfo(f"Output directory: {output_dir}")

        with Pool(num_workers) as pool:
            for image in frame_iter:
                batch_args.append((frame_num, image, output_dir))
                if len(batch_args) >= batch_size:
                    batch_results = pool.map(process_frame, batch_args)
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
