import numpy as np

def map_direction(angle):
    """Map Canny edge direction to NSST direction index (0-7)"""
    bin_edges = np.linspace(0, 180, 9)
    return np.digitize(angle, bin_edges) % 8

def reconstruct(low_freq, denoised_bands):
    """Reconstruct image from NSST components"""
    # Flatten directional bands
    high_coeffs = np.dstack([band for band in denoised_bands])
    coeffs = np.dstack([high_coeffs, low_freq[..., None]])
    
    # Inverse NSST (pseudo-code)
    n = low_freq.shape[0]
    shearlet_system = shearlab.getshearletsystem2D(0, n, n, len(denoised_bands))
    return shearlab.shearrec2D(coeffs, shearlet_system)
