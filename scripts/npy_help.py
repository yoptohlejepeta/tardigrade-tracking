import numpy as np
from pathlib import Path


def compress_npy_directory(directory_path, output_filename="compressed_data.npz"):
    """Loads all .npy files in a directory and saves them into a single compressed .npz file."""

    npy_files = Path(directory_path).glob("*.npy")
    arrays = {file.stem: np.load(file) for file in npy_files}

    if not arrays:
        print("No .npy files found in the directory.")
        return

    np.savez_compressed(output_filename, **arrays)
    print(f"Successfully compressed {len(arrays)} arrays into '{output_filename}'.")


compress_npy_directory("./dataset_1/T17.24Reh/", "./dataset_1/T17.24Reh.npz")
compress_npy_directory("./dataset_1/T17.48Reh/", "./dataset_1/T17.48Reh.npz")
compress_npy_directory("./dataset_1/T17.72Reh/", "./dataset_1/T17.72Reh.npz")
compress_npy_directory("./dataset_1/T18.24Reh/", "./dataset_1/T18.24Reh.npz")
compress_npy_directory("./dataset_1/T18.48Reh/", "./dataset_1/T18.48Reh.npz")
compress_npy_directory("./dataset_1/T18.72Reh/", "./dataset_1/T18.72Reh.npz")
