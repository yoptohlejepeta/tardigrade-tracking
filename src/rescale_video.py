import cv2
import numpy as np
import argparse
from skimage.exposure import rescale_intensity
from pathlib import Path
from rich.progress import Progress
import concurrent.futures
import multiprocessing
from collections import deque


def rescale_image(image):
    cropped = image[:, 300:1500]

    if len(cropped.shape) == 3:
        b, g, r = cropped[:, :, 0], cropped[:, :, 1], cropped[:, :, 2]
        gray = 1 / 3 * r + 1 / 3 * g + 1 / 3 * b
    else:
        gray = cropped

    gray_rescaled = rescale_intensity(gray, in_range=(70, 200), out_range=(0, 255))
    return gray_rescaled.astype(np.uint8)


def process_video(input_path, output_path, workers=None):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)

    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist")
        return

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error opening video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = cap.read()
    if not ret:
        print("Could not read first frame")
        return

    processed = rescale_image(frame)
    height, width = processed.shape

    print(f"Processing: {input_path}")
    print(f"Output to: {output_path}")
    print(f"Workers: {workers}")
    print(
        f"Video Info: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} -> {width}x{height}"
    )
    print(f"FPS: {fps}, Total Frames: {total_frames}")

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    except AttributeError:
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=False)

    if not out.isOpened():
        print(
            "Error: Could not create video writer. Trying with isColor=True and converting frames..."
        )
        out = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height), isColor=True
        )
        force_color = True
    else:
        force_color = False

    frame_out = processed
    if force_color:
        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_GRAY2BGR)
    out.write(frame_out)

    count = 1
    max_pending = workers * 2
    futures = deque()
    reading_active = True

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing video...", total=total_frames)
            progress.update(task, advance=1)

            while futures or reading_active:
                while reading_active and len(futures) < max_pending:
                    ret, frame = cap.read()
                    if not ret:
                        reading_active = False
                        break

                    future = executor.submit(rescale_image, frame)
                    futures.append(future)

                if not futures:
                    break

                try:
                    frame_out = futures.popleft().result()

                    if force_color:
                        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_GRAY2BGR)

                    out.write(frame_out)

                    count += 1
                    progress.update(task, advance=1)
                except Exception as e:
                    print(f"\nError processing frame: {e}")
                    break

    print(f"\nFinished processing {count} frames.")
    cap.release()
    out.release()


def main():
    parser = argparse.ArgumentParser(
        description="Rescale video intensity with multiprocessing."
    )
    parser.add_argument("input", type=str, help="Path to input video")
    parser.add_argument("output", type=str, help="Path to output video")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)",
    )

    args = parser.parse_args()

    process_video(args.input, args.output, args.workers)


if __name__ == "__main__":
    main()
