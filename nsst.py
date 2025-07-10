import shearlab
import numpy as np

def decompose(image, scales=3):
    n = image.shape[0]
    shearlet_system = shearlab.getshearletsystem2D(0, n, n, scales)
    coeffs = shearlab.sheardec2D(image, shearlet_system)
    
    # Split coefficients
    n_shearlets = shearlet_system['nShearlets']
    low_freq = coeffs[:, :, -1]
    high_freq = coeffs[:, :, :-1]
    
    # Group by scale (assume order: scale0, scale1, scale2)
    bands = []
    start = 0
    for j in range(scales):
        n_dir = 2 ** (j + 1)
        bands.append(high_freq[:, :, start:start+n_dir])
        start += n_dir
    return low_freq, bands
