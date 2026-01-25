import os
from pathlib import Path
from typing import Annotated

import imageio.v3 as iio
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.info import loginfo


class Arguments(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)
    input_path: Annotated[
        Path,
        Field(
            title="Input directory with videos",
            validation_alias=AliasChoices("i", "input_path"),
        ),
    ]
    output_path: Annotated[
        str,
        Field(
            title="Output directory",
            validation_alias=AliasChoices("o", "output_dir"),
        ),
    ] = "frames"
    pattern: Annotated[
        str,
        Field(
            title="Pattern to match video files",
            validation_alias=AliasChoices("p", "pattern"),
        ),
    ] = "*.mp4"
    frame_step: Annotated[
        int,
        Field(
            title="Process every nth frame",
            validation_alias=AliasChoices("s", "frame_step"),
        ),
    ] = 1
    format: Annotated[
        str,
        Field(
            title="Image format (png or jpg)",
            validation_alias=AliasChoices("f", "format"),
        ),
    ] = "png"


def main():
    args = Arguments()  # pyright: ignore[reportCallIssue]

    if not args.input_path.exists():
        loginfo(f"Error: Input directory not found: {args.input_path}")
        return

    loginfo(f"Input directory: {args.input_path}")
    loginfo(f"Pattern: {args.pattern}")
    loginfo(f"Frame step: {args.frame_step}")

    for file in args.input_path.glob(args.pattern):
        if not file.is_file():
            continue

        output_dir = f"{args.output_path}/{file.stem}"
        os.makedirs(output_dir, exist_ok=True)
        loginfo(f"Processing video: {file}")
        loginfo(f"Output directory: {output_dir}")

        frame_iter = iio.imiter(file)
        frame_num = 0
        saved_count = 0

        for image in frame_iter:
            if frame_num % args.frame_step == 0:
                output_path = f"{output_dir}/frame_{frame_num:06d}.{args.format}"
                iio.imwrite(output_path, image)
                saved_count += 1
                if saved_count % 100 == 0:
                    loginfo(
                        f"  Saved {saved_count} frames (processed {frame_num} total)"
                    )
            frame_num += 1

        loginfo(
            f"Done with {file.stem}: processed {frame_num} total frames, saved {saved_count} frames"
        )


if __name__ == "__main__":
    main()
