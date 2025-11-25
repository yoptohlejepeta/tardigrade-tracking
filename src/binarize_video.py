import cv2
import numpy as np
import argparse
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen
from skimage.morphology import opening, disk, remove_small_objects
from pathlib import Path
from rich.progress import Progress
import concurrent.futures
import multiprocessing
from collections import deque


def binarize_image(image):
    # Crop the image as requested
    cropped = image[:, 300:1500]

    # Extract channels (Assuming image is BGR from cv2, but averaging makes channel order irrelevant)
    # Using cropped image for processing
    if len(cropped.shape) == 3:
        b, g, r = cropped[:, :, 0], cropped[:, :, 1], cropped[:, :, 2]
        # Note: User snippet had r assigned to index 0, but formula is symmetric average
        gray = 1 / 3 * r + 1 / 3 * g + 1 / 3 * b
    else:
        gray = cropped

    # Rescale intensity
    gray_rescaled = rescale_intensity(gray, in_range=(70, 200), out_range=(0, 255))

    # Yen Thresholding
    yenThresh = threshold_yen(gray_rescaled)
    binaryYen = gray_rescaled > yenThresh

    # Morphological operations
    mask_size = 5
    binaryCleaned = binaryYen
    binaryCleaned = opening(binaryCleaned, footprint=disk(mask_size))
    binaryCleaned = remove_small_objects(binaryCleaned, min_size=600)

    return binaryCleaned


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

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame to determine output size
    ret, frame = cap.read()
    if not ret:
        print("Could not read first frame")
        return

    # Process first frame locally to get dimensions and verify it works
    processed = binarize_image(frame)
    height, width = processed.shape

    print(f"Processing: {input_path}")
    print(f"Output to: {output_path}")
    print(f"Workers: {workers}")
    print(
        f"Video Info: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} -> {width}x{height}"
    )
    print(f"FPS: {fps}, Total Frames: {total_frames}")

    # Initialize VideoWriter
    # Using isColor=False for grayscale output
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

    # Write first frame
    frame_out = (processed * 255).astype(np.uint8)
    if force_color:
        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_GRAY2BGR)
    out.write(frame_out)

    count = 1

    # Setup multiprocessing
    # Limit queue size to prevent memory exhaustion
    max_pending = workers * 2
    futures = deque()
    reading_active = True

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing video...", total=total_frames)
            progress.update(task, advance=1)  # First frame done

            while futures or reading_active:
                # Fill the queue
                while reading_active and len(futures) < max_pending:
                    ret, frame = cap.read()
                    if not ret:
                        reading_active = False
                        break

                    # Submit task
                    future = executor.submit(binarize_image, frame)
                    futures.append(future)

                if not futures:
                    break

                # Retrieve results in order
                # popleft() gets the oldest future (FIFO)
                # result() blocks until it's ready
                try:
                    binary_mask = futures.popleft().result()

                    # Convert boolean to uint8
                    frame_out = (binary_mask * 255).astype(np.uint8)

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
        description="Binarize video using skimage filters with multiprocessing."
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
