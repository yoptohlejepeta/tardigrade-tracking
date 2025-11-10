import tifffile
import imageio.v2 as imageio
from pathlib import Path
import numpy as np

png_files = sorted(Path("images/Nocodazole/N21.tuns/").glob("*.png"))
print(f"Loading {len(png_files)} files...")

images = [imageio.imread(str(f)) for f in png_files]
stack = np.array(images)
print(f"Stack shape: {stack.shape}")

tifffile.imwrite("tif_videos/Nocodazole/N21.tuns/N21.tuns.tif", 
                 stack, 
                 compression='lzw')
print("Done!")
