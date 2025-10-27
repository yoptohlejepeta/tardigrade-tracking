import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from ultralytics import SAM
from ultralytics.models.sam import SAM2VideoPredictor
import torch
import cv2
from pathlib import Path
import gc

# Configure CUDA memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# Memory management settings
MAX_FRAMES_PER_BATCH = 100  # Process video in chunks
REDUCE_OBJECTS = True  # Track only most prominent objects
MAX_OBJECTS = 10  # Maximum objects to track (reduce if still OOM)
IMAGE_SIZE = 512  # Reduce from 1024 to save memory

# Path to your video
input_path = "data/C4.tuns.mkv"

# Get video info
print("Checking video...")
cap = cv2.VideoCapture(input_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

print(f"Video: {total_frames} frames, {width}x{height}, {fps:.2f} FPS")
print(f"Will process in batches of {MAX_FRAMES_PER_BATCH} frames")

# Step 1: Detect objects in the first frame using SAM
print("\nDetecting objects in first frame...")

# Extract only the first frame
cap = cv2.VideoCapture(input_path)
ret, first_frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read first frame")
    exit(1)

# Save first frame temporarily
temp_first_frame = "temp_first_frame.jpg"
cv2.imwrite(temp_first_frame, first_frame)

# Run SAM on just the first frame
model = SAM("sam2.1_b.pt")
model.to(device="cuda")
results = model(temp_first_frame)

# Clean up temp file
os.unlink(temp_first_frame)

# Extract masks and compute centroids with sizes
masks = results[0].masks.data.cpu().numpy()
centroids_with_size = []

for i, mask in enumerate(masks):
    binary_mask = (mask > 0.5).astype(np.uint8)
    mask_area = binary_mask.sum()
    cy, cx = center_of_mass(binary_mask)
    centroids_with_size.append({
        'id': i,
        'centroid': [int(cx), int(cy)],
        'area': mask_area
    })

# Sort by area (largest first) and select top objects
centroids_with_size.sort(key=lambda x: x['area'], reverse=True)

if REDUCE_OBJECTS and len(centroids_with_size) > MAX_OBJECTS:
    print(f"\nReducing from {len(centroids_with_size)} to {MAX_OBJECTS} largest objects to save memory")
    centroids_with_size = centroids_with_size[:MAX_OBJECTS]

centroids = [obj['centroid'] for obj in centroids_with_size]

print(f"\nTracking {len(centroids)} objects:")
if len(centroids) > 15:
    print(f"⚠️  WARNING: Tracking {len(centroids)} objects may cause CUDA OOM with GTX 1650 (3.7GB VRAM)")
    print(f"   Consider setting REDUCE_OBJECTS=True and MAX_OBJECTS=10 if you encounter errors")
for i, obj in enumerate(centroids_with_size):
    cx, cy = obj['centroid']
    print(f"  Object {i+1}: Centroid ({cx}, {cy}), Area: {obj['area']} pixels")

# Visualize first frame with selected centroids
plotted_image = results[0].plot(masks=True, boxes=False)
plt.figure(figsize=(12, 8))
plt.imshow(plotted_image)
plt.axis('off')
plt.title(f"Selected Objects: {len(centroids)} (largest by area)")
for i, [cx, cy] in enumerate(centroids):
    plt.scatter(cx, cy, color='red', s=100, marker='o', edgecolors='white', linewidth=2)
    plt.text(cx, cy-15, f'{i+1}', color='yellow', fontsize=12, ha='center', 
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
plt.tight_layout()
plt.savefig('detected_objects_frame1.png', dpi=150, bbox_inches='tight')
plt.show()

# Clear memory
del model, results, masks
torch.cuda.empty_cache()
gc.collect()

# Prepare colors for visualization
colors = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))

# Step 2: Process video in batches
print("\n" + "="*60)
print("PROCESSING VIDEO IN BATCHES")
print("="*60)

# Create output directories
output_dir = Path("tracking_output")
masks_dir = output_dir / "masks"
images_dir = output_dir / "images"
checkpoints_dir = output_dir / "checkpoints"

for d in [output_dir, masks_dir, images_dir, checkpoints_dir]:
    d.mkdir(exist_ok=True)

print(f"Output directory: {output_dir}")
print(f"Images will be saved to: {images_dir}")

all_trajectories = {i: [] for i in range(len(centroids))}
num_batches = (total_frames + MAX_FRAMES_PER_BATCH - 1) // MAX_FRAMES_PER_BATCH

# Check for existing checkpoint
checkpoint_file = checkpoints_dir / "progress.npz"
start_batch = 0

if checkpoint_file.exists():
    print(f"\n📌 Found existing checkpoint. Loading...")
    checkpoint = np.load(checkpoint_file, allow_pickle=True)
    checkpoint_centroids = checkpoint['centroids_initial']
    
    # Check if number of objects matches
    if len(checkpoint_centroids) == len(centroids):
        all_trajectories = checkpoint['trajectories'].item()
        start_batch = int(checkpoint['last_batch']) + 1
        print(f"Resuming from batch {start_batch + 1}/{num_batches}")
    else:
        print(f"⚠️  Checkpoint has {len(checkpoint_centroids)} objects but detected {len(centroids)} objects.")
        print("Starting fresh (checkpoint deleted)")
        os.unlink(checkpoint_file)
        start_batch = 0

for batch_idx in range(start_batch, num_batches):
    start_frame = batch_idx * MAX_FRAMES_PER_BATCH
    end_frame = min(start_frame + MAX_FRAMES_PER_BATCH, total_frames)
    
    print(f"\n--- Batch {batch_idx + 1}/{num_batches}: Frames {start_frame}-{end_frame-1} ---")
    
    # Extract frames for this batch
    print("Extracting frames...")
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    batch_frames = []
    for i in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        batch_frames.append(frame)
    cap.release()
    
    print(f"Extracted {len(batch_frames)} frames")
    
    # Create temporary video for this batch
    temp_video = f"temp_batch_{batch_idx}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    for frame in batch_frames:
        out.write(frame)
    out.release()
    
    # Run SAM2VideoPredictor on this batch
    print("Initializing predictor...")
    overrides = dict(
        conf=0.25, 
        task="segment", 
        mode="predict", 
        imgsz=IMAGE_SIZE,  # Reduced size
        model="sam2.1_b.pt",
        device="cuda",
        verbose=False
    )
    
    try:
        predictor = SAM2VideoPredictor(overrides=overrides)
        
        # Use bboxes from first batch, or last known positions for subsequent batches
        if batch_idx == 0:
            prompt_bboxes = bboxes
        else:
            # Calculate bboxes from last known positions in previous batch
            prompt_bboxes = []
            for obj_id in range(len(centroids)):
                if all_trajectories[obj_id]:
                    # Get last known position
                    last_pos = all_trajectories[obj_id][-1]
                    last_cx, last_cy = int(last_pos[1]), int(last_pos[2])
                    
                    # Estimate bbox size from original bbox
                    orig_bbox = bboxes[obj_id]
                    bbox_width = orig_bbox[2] - orig_bbox[0]
                    bbox_height = orig_bbox[3] - orig_bbox[1]
                    
                    # Create bbox centered at last known position
                    new_bbox = [
                        max(0, last_cx - bbox_width // 2),
                        max(0, last_cy - bbox_height // 2),
                        min(width, last_cx + bbox_width // 2),
                        min(height, last_cy + bbox_height // 2)
                    ]
                    prompt_bboxes.append(new_bbox)
                else:
                    prompt_bboxes.append(bboxes[obj_id])
        
        print(f"Tracking {len(prompt_bboxes)} objects with bounding boxes...")
        results = predictor(
            source=temp_video,
            bboxes=prompt_bboxes,  # Use bboxes instead of points
            stream=True  # Use streaming to save memory
        )
        
        # Process results and save masks
        for local_frame_idx, result in enumerate(results):
            global_frame_idx = start_frame + local_frame_idx
            
            if result.masks is not None:
                frame_masks = result.masks.data.cpu().numpy()
                
                # Save masks for this frame
                frame_data = {
                    'frame_idx': global_frame_idx,
                    'masks': frame_masks,
                    'centroids': []
                }
                
                # Get original frame
                original_frame = batch_frames[local_frame_idx]
                
                # Create visualization
                vis_frame = original_frame.copy()
                colors_rgb = (colors * 255).astype(np.uint8)
                
                for obj_id, mask in enumerate(frame_masks):
                    if obj_id >= len(centroids):
                        break
                    
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    if binary_mask.sum() > 0:
                        cy, cx = center_of_mass(binary_mask)
                        all_trajectories[obj_id].append((global_frame_idx, cx, cy))
                        frame_data['centroids'].append((obj_id, cx, cy))
                        
                        # Overlay mask on image
                        color = colors_rgb[obj_id]
                        mask_overlay = np.zeros_like(vis_frame)
                        mask_overlay[binary_mask > 0] = color[:3]
                        vis_frame = cv2.addWeighted(vis_frame, 1.0, mask_overlay, 0.4, 0)
                        
                        # Draw centroid
                        cv2.circle(vis_frame, (int(cx), int(cy)), 5, color[:3].tolist(), -1)
                        cv2.circle(vis_frame, (int(cx), int(cy)), 7, (255, 255, 255), 2)
                        
                        # Draw object ID
                        cv2.putText(vis_frame, f'{obj_id+1}', 
                                  (int(cx)+10, int(cy)-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                  (255, 255, 255), 2)
                
                # Save visualization image (respecting SAVE_EVERY_N_FRAMES)
                if global_frame_idx % SAVE_EVERY_N_FRAMES == 0:
                    cv2.imwrite(
                        str(images_dir / f"frame_{global_frame_idx:06d}.jpg"),
                        vis_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 90]
                    )
                
                # Always save masks data
                np.savez_compressed(
                    masks_dir / f"frame_{global_frame_idx:06d}.npz",
                    **frame_data
                )
            
            if (local_frame_idx + 1) % 20 == 0:
                print(f"  Processed {local_frame_idx + 1}/{len(batch_frames)} frames")
        
        print(f"Batch {batch_idx + 1} complete")
        
        # Save checkpoint after each batch
        np.savez(
            checkpoint_file,
            trajectories=all_trajectories,
            last_batch=batch_idx,
            centroids_initial=centroids,
            bboxes_initial=bboxes
        )
        print(f"💾 Checkpoint saved")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Saving progress...")
        np.savez(
            checkpoint_file,
            trajectories=all_trajectories,
            last_batch=batch_idx - 1,  # Save previous batch as last completed
            centroids_initial=centroids,
            bboxes_initial=bboxes
        )
        print(f"Progress saved. Run again to resume from batch {batch_idx + 1}")
        break
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n⚠️  CUDA OOM in batch {batch_idx + 1}. Try reducing MAX_OBJECTS or IMAGE_SIZE further.")
        print(f"Error: {e}")
        # Save checkpoint before breaking
        np.savez(
            checkpoint_file,
            trajectories=all_trajectories,
            last_batch=batch_idx - 1,
            centroids_initial=centroids,
            bboxes_initial=bboxes
        )
        break
    finally:
        # Clean up
        del predictor
        if 'results' in locals():
            del results
        torch.cuda.empty_cache()
        gc.collect()
        
        # Remove temporary video
        if os.path.exists(temp_video):
            os.unlink(temp_video)

print("\n" + "="*60)
print("TRACKING COMPLETE")
print("="*60)

# Step 3: Visualize trajectories
print("\nCreating trajectory visualization...")
cap = cv2.VideoCapture(input_path)
ret, first_frame = cap.read()
first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
cap.release()

plt.figure(figsize=(14, 10))
colors = plt.cm.rainbow(np.linspace(0, 1, len(all_trajectories)))

for (obj_id, trajectory), color in zip(all_trajectories.items(), colors):
    if trajectory:
        frames_list, x_coords, y_coords = zip(*trajectory)
        plt.plot(x_coords, y_coords, '-', color=color, label=f'Object {obj_id + 1}', 
                 linewidth=2, alpha=0.7)
        # Mark start and end
        plt.scatter(x_coords[0], y_coords[0], color=color, s=200, marker='*', 
                   edgecolors='black', linewidth=2, zorder=5)
        plt.scatter(x_coords[-1], y_coords[-1], color=color, s=200, marker='s', 
                   edgecolors='black', linewidth=2, zorder=5)

plt.imshow(first_frame_rgb, alpha=0.3)
plt.legend(loc='best', fontsize=8, ncol=2)
plt.title(f'Object Trajectories ({len(all_trajectories)} objects)\n(★ = start, ■ = end)')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('object_trajectories.png', dpi=150, bbox_inches='tight')
plt.show()

# Step 4: Export trajectory data
print("\nExporting trajectory data...")
trajectory_data = {
    'centroids_initial': centroids,
    'trajectories': all_trajectories,
    'num_objects': len(centroids),
    'total_frames': total_frames,
    'video_info': {
        'width': width,
        'height': height,
        'fps': fps
    }
}
np.save('object_trajectories.npy', trajectory_data, allow_pickle=True)

# Create CSV export
print("Creating CSV export...")
with open('trajectories.csv', 'w') as f:
    f.write('object_id,frame,x,y\n')
    for obj_id, trajectory in all_trajectories.items():
        for frame_idx, cx, cy in trajectory:
            f.write(f'{obj_id+1},{frame_idx},{cx:.2f},{cy:.2f}\n')

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Tracked {len(centroids)} objects")
print(f"Saved images in: {images_dir}")
print(f"Total image files: {len(list(images_dir.glob('*.jpg')))}")
print(f"Saved mask data in: {masks_dir}")
print(f"Total mask files: {len(list(masks_dir.glob('*.npz')))}")
print("\nTrajectory Summary:")
for obj_id, trajectory in all_trajectories.items():
    if trajectory:
        frames_tracked = len(trajectory)
        print(f"  Object {obj_id + 1}: {frames_tracked} frames tracked")
        if trajectory:
            start_x, start_y = trajectory[0][1], trajectory[0][2]
            end_x, end_y = trajectory[-1][1], trajectory[-1][2]
            displacement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            print(f"    Total displacement: {displacement:.2f} pixels")

print("\nSaved files:")
print(f"  - {output_dir}/")
print(f"    ├── images/ ({len(list(images_dir.glob('*.jpg')))} visualization images)")
print(f"    ├── masks/ ({len(list(masks_dir.glob('*.npz')))} mask data files)")
print(f"    └── checkpoints/progress.npz (for resuming)")
print("  - detected_objects_frame1.png")
print("  - object_trajectories.png")
print("  - object_trajectories.npy")
print("  - trajectories.csv")

torch.cuda.empty_cache()
print("\nDone!")
