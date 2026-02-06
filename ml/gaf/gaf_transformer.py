"""
Gramian Angular Field (GAF) transformation functions.

Implements GASF (Summation) and GADF (Difference) transformations
for converting time series to 2D images.

References:
- Wang & Oates (2015): "Imaging Time-Series to Improve Classification and Imputation"
- IEEE/CAA (2020): "Deep Learning and Time Series-to-Image Encoding for Financial Forecasting"
"""

import numpy as np
from typing import Tuple, Optional
from skimage.transform import resize


def normalize_timeseries(ts: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize time series to [-1, 1] range.

    Args:
        ts: Time series array of shape (n,)
        method: Normalization method ('minmax' or 'zscore')

    Returns:
        Normalized time series in [-1, 1]
    """
    if method == 'minmax':
        ts_min = ts.min()
        ts_max = ts.max()

        if ts_max - ts_min < 1e-8:
            # Constant time series
            return np.zeros_like(ts)

        # Normalize to [0, 1]
        ts_norm = (ts - ts_min) / (ts_max - ts_min)

        # Scale to [-1, 1]
        ts_scaled = 2 * ts_norm - 1

        return ts_scaled

    elif method == 'zscore':
        ts_mean = ts.mean()
        ts_std = ts.std()

        if ts_std < 1e-8:
            return np.zeros_like(ts)

        # Z-score normalization
        ts_norm = (ts - ts_mean) / ts_std

        # Clip to [-3, 3] and scale to [-1, 1]
        ts_clipped = np.clip(ts_norm, -3, 3)
        ts_scaled = ts_clipped / 3

        return ts_scaled

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def polar_encode(ts_normalized: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode normalized time series into polar coordinates.

    Args:
        ts_normalized: Normalized time series in [-1, 1]

    Returns:
        Tuple of (phi, r) where:
        - phi: Angular values (radians)
        - r: Radial values (normalized timestamps)
    """
    # Clip to [-1, 1] to avoid arccos domain errors
    ts_clipped = np.clip(ts_normalized, -1, 1)

    # Angular encoding: phi = arccos(x)
    phi = np.arccos(ts_clipped)

    # Radial encoding: r = t / N (normalized timestamp)
    n = len(ts_normalized)
    r = np.arange(n) / n

    return phi, r


def generate_gasf(ts: np.ndarray, size: Optional[int] = None) -> np.ndarray:
    """
    Generate Gramian Angular Summation Field (GASF).

    GASF[i,j] = cos(phi_i + phi_j)

    Args:
        ts: Time series array of shape (n,)
        size: Target image size (will resize if specified)

    Returns:
        GASF matrix of shape (n, n) or (size, size) if size specified
    """
    # Normalize
    ts_norm = normalize_timeseries(ts)

    # Polar encode
    phi, _ = polar_encode(ts_norm)

    # GASF: cos(phi_i + phi_j)
    # Efficient computation: cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    gasf = np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)

    # Resize if requested
    if size is not None and size != len(ts):
        gasf = resize(gasf, (size, size), anti_aliasing=True, preserve_range=True)

    return gasf


def generate_gadf(ts: np.ndarray, size: Optional[int] = None) -> np.ndarray:
    """
    Generate Gramian Angular Difference Field (GADF).

    GADF[i,j] = sin(phi_i - phi_j)

    Args:
        ts: Time series array of shape (n,)
        size: Target image size (will resize if specified)

    Returns:
        GADF matrix of shape (n, n) or (size, size) if size specified
    """
    # Normalize
    ts_norm = normalize_timeseries(ts)

    # Polar encode
    phi, _ = polar_encode(ts_norm)

    # GADF: sin(phi_i - phi_j)
    # Efficient computation: sin(a-b) = sin(a)cos(b) - cos(a)sin(b)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    gadf = np.outer(sin_phi, cos_phi) - np.outer(cos_phi, sin_phi)

    # Resize if requested
    if size is not None and size != len(ts):
        gadf = resize(gadf, (size, size), anti_aliasing=True, preserve_range=True)

    return gadf


def generate_gaf(ts: np.ndarray, size: int = 64, mode: str = 'both') -> np.ndarray:
    """
    Generate GAF image (GASF and/or GADF).

    Args:
        ts: Time series array of shape (n,)
        size: Target image size (default 64x64)
        mode: 'gasf', 'gadf', or 'both' (default)

    Returns:
        GAF image of shape:
        - (size, size) if mode is 'gasf' or 'gadf'
        - (size, size, 2) if mode is 'both' (GASF in channel 0, GADF in channel 1)
    """
    if mode == 'gasf':
        gasf = generate_gasf(ts, size=size)
        return gasf

    elif mode == 'gadf':
        gadf = generate_gadf(ts, size=size)
        return gadf

    elif mode == 'both':
        gasf = generate_gasf(ts, size=size)
        gadf = generate_gadf(ts, size=size)

        # Stack as 2-channel image
        gaf_image = np.stack([gasf, gadf], axis=-1)
        return gaf_image

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'gasf', 'gadf', or 'both'.")


def extract_windows(df, window_size: int, feature: str = 'close'):
    """
    Extract rolling windows from dataframe.

    Args:
        df: DataFrame with OHLCV data
        window_size: Window size in minutes
        feature: Feature to extract (default 'close')

    Returns:
        List of windows as numpy arrays
    """
    windows = []
    values = df[feature].values

    for i in range(len(values) - window_size + 1):
        window = values[i:i+window_size]
        windows.append(window)

    return windows


def generate_gaf_from_df(df, window_size: int, size: int = 64, feature: str = 'close', mode: str = 'both'):
    """
    Generate GAF images from DataFrame.

    Args:
        df: DataFrame with OHLCV data
        window_size: Window size in minutes
        size: Target image size
        feature: Feature to extract
        mode: 'gasf', 'gadf', or 'both'

    Returns:
        List of GAF images as numpy arrays
    """
    windows = extract_windows(df, window_size, feature)

    gaf_images = []
    for window in windows:
        gaf = generate_gaf(window, size=size, mode=mode)
        gaf_images.append(gaf)

    return np.array(gaf_images)


# Efficiency utilities

def batch_generate_gaf(windows: list, size: int = 64, mode: str = 'both', n_jobs: int = -1):
    """
    Generate GAF images for multiple windows in parallel.

    Args:
        windows: List of time series windows
        size: Target image size
        mode: 'gasf', 'gadf', or 'both'
        n_jobs: Number of parallel jobs (-1 for all CPUs)

    Returns:
        Array of GAF images
    """
    from joblib import Parallel, delayed

    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()

    gaf_images = Parallel(n_jobs=n_jobs)(
        delayed(generate_gaf)(window, size, mode)
        for window in windows
    )

    return np.array(gaf_images)


if __name__ == '__main__':
    # Test GAF transformation
    import matplotlib.pyplot as plt

    # Generate synthetic time series
    t = np.linspace(0, 4*np.pi, 60)
    ts = np.sin(t) + 0.1 * np.random.randn(60)

    # Generate GAF
    gasf = generate_gasf(ts, size=64)
    gadf = generate_gadf(ts, size=64)
    gaf = generate_gaf(ts, size=64, mode='both')

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(gasf, cmap='rainbow', origin='lower')
    axes[0].set_title('GASF')
    axes[0].axis('off')

    axes[1].imshow(gadf, cmap='rainbow', origin='lower')
    axes[1].set_title('GADF')
    axes[1].axis('off')

    axes[2].plot(ts)
    axes[2].set_title('Original Time Series')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('gaf_test.png')
    print("✅ GAF transformation test complete. Saved to gaf_test.png")
