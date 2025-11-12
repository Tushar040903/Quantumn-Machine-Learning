import numpy as np
import os

def amplitude_encode(image: np.ndarray) -> np.ndarray:
    """
    Quantum-inspired amplitude encoding for a single image.
    Converts an image into a normalized amplitude vector.
    """
    # Flatten the image (28x28 -> 784)
    flat = image.flatten().astype(np.float32)

    # Normalize pixel intensities (L2 norm)
    norm = np.linalg.norm(flat)
    if norm == 0:
        return flat  # avoid division by zero
    return flat / norm


def encode_dataset(x_data: np.ndarray, save_path: str = None) -> np.ndarray:
    """
    Encode all images in the dataset using amplitude encoding.
    Optionally saves the encoded data as .npy file.
    """
    print(f"Encoding {len(x_data)} images ...")
    encoded = np.array([amplitude_encode(img) for img in x_data])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, encoded)
        print(f"✅ Encoded dataset saved at: {save_path}")

    return encoded
