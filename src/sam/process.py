import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import SAM
from ultralytics.models.sam import SAM2VideoPredictor
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    # MPS is for Mac, but let's stick to cuda/cpu for now as environment seems to be linux
    return "cpu"


def process_image(input_path, output_path, model_path, max_dim=1024):
    print(f"Processing image: {input_path}")
    device = get_device()

    if device == "cuda":
        torch.cuda.empty_cache()

    model = SAM(model_path)
    model.to(device=device)

    # Run inference with limited image size
    results = model(
        input_path,
        imgsz=max_dim,
        conf=0.1,
        iou=0.3,
    )

    # Plot results
    # results[0].plot() returns a numpy array (BGR)
    plotted_image = results[0].plot(labels=False, boxes=False)

    # Save result
    cv2.imwrite(str(output_path), plotted_image)
    print(f"Saved segmented image to {output_path}")


def process_video(input_path, output_path, model_path, max_dim=1024):
    print(f"Processing video: {input_path}")
    device = get_device()

    # 1. Analyze first frame to find objects to track
    cap = cv2.VideoCapture(str(input_path))
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame of video")

    # Downscale first frame if too large (saves GPU memory)
    orig_h, orig_w = first_frame.shape[:2]
    scale = 1.0
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        first_frame_resized = cv2.resize(first_frame, (new_w, new_h))
        print(
            f"Resized first frame from {orig_w}x{orig_h} to {new_w}x{new_h} for detection"
        )
    else:
        first_frame_resized = first_frame

    # Save first frame temporarily for SAM
    temp_first_frame = "temp_first_frame.jpg"
    cv2.imwrite(temp_first_frame, first_frame_resized)

    print("Detecting objects in first frame...")
    model = SAM(model_path)
    model.to(device=device)

    # Clear cache before inference
    if device == "cuda":
        torch.cuda.empty_cache()

    results = model(temp_first_frame, imgsz=max_dim)

    # Clean up temp file
    if os.path.exists(temp_first_frame):
        os.unlink(temp_first_frame)

    if not results[0].masks:
        print("No objects detected in first frame.")
        cap.release()
        return

    # Extract masks and compute centroids
    masks = results[0].masks.data.cpu().numpy()  # [N, H, W]

    # Filter small masks or prepare prompts
    points = []
    labels = []

    print(f"Found {len(masks)} objects in first frame.")

    for i, mask in enumerate(masks):
        binary_mask = (mask > 0.5).astype(np.uint8)
        if binary_mask.sum() == 0:
            continue
        cy, cx = center_of_mass(binary_mask)
        # Scale coordinates back to original resolution
        points.append([int(cx / scale), int(cy / scale)])
        labels.append(1)  # 1 = foreground

    if not points:
        print("No valid objects found to track.")
        cap.release()
        return

    # 2. Setup Video Predictor
    del model
    torch.cuda.empty_cache()

    cap.release()  # Release to reopen or just get properties

    # Get video properties
    cap = cv2.VideoCapture(str(input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    output_video_path = str(output_path)
    # Ensure output ends with .mp4 or .avi
    if not (
        output_video_path.endswith(".mp4")
        or output_video_path.endswith(".avi")
        or output_video_path.endswith(".mkv")
    ):
        output_video_path += ".mp4"

    print(f"Initializing tracking for {len(points)} objects...")

    overrides = dict(
        task="segment",
        mode="predict",
        model=model_path,
        device=device,
        imgsz=max_dim,
    )
    predictor = SAM2VideoPredictor(overrides=overrides)

    # Generate colors
    colors = np.random.randint(0, 255, size=(len(points), 3), dtype=np.uint8)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Run inference
    results_generator = predictor(
        source=str(input_path), points=points, labels=labels, stream=True
    )

    print(f"Processing {total_frames} frames...")

    frame_count = 0
    for result in results_generator:
        frame = result.orig_img

        if result.masks is not None:
            masks_data = result.masks.data.cpu().numpy()
            # Track IDs should correspond to the input order if persistent
            # But SAM2 output usually includes track_ids

            track_ids = None
            if (
                result.boxes is not None
                and hasattr(result.boxes, "id")
                and result.boxes.id is not None
            ):
                track_ids = result.boxes.id.int().cpu().tolist()
            else:
                # Fallback if no IDs provided (might happen in some modes), assume order?
                # Usually SAM2 returns IDs. If not, we might just enumerate masks.
                track_ids = range(len(masks_data))

            # Draw masks
            for i, mask in enumerate(masks_data):
                idx = track_ids[i] if i < len(track_ids) else i
                color = colors[idx % len(colors)].tolist()

                binary_mask = (mask > 0.5).astype(np.uint8)

                # Create colored overlay
                overlay = np.zeros_like(frame)
                overlay[binary_mask == 1] = color

                frame = cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)

        out.write(frame)
        frame_count += 1
        if frame_count % 20 == 0:
            print(f"Processed {frame_count} frames...")

        # Periodically clear GPU cache to prevent OOM
        if frame_count % 100 == 0 and device == "cuda":
            torch.cuda.empty_cache()

    out.release()
    print(f"Saved segmented video to {output_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Segment images or videos using SAM.")
    parser.add_argument("input", type=str, help="Path to input image or video")
    parser.add_argument("--output", type=str, default=None, help="Path to output file")
    parser.add_argument(
        "--model", type=str, default="sam2.1_b.pt", help="Path to SAM model checkpoint"
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1024,
        help="Max dimension for first frame detection (reduces GPU memory)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        # Default output name
        suffix = input_path.suffix
        stem = input_path.stem
        if suffix.lower() in [".mp4", ".avi", ".mkv", ".mov"]:
            output_path = input_path.with_name(f"{stem}_segmented.mp4")
        else:
            output_path = input_path.with_name(f"{stem}_segmented{suffix}")

    # Determine if image or video
    # Simple extension check
    video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv"]
    is_video = input_path.suffix.lower() in video_extensions

    try:
        if is_video:
            process_video(input_path, output_path, args.model, args.max_dim)
        else:
            process_image(input_path, output_path, args.model, args.max_dim)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
