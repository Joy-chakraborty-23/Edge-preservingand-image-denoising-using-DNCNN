import numpy as np
from .nsst import decompose
from .patch_extractor import extract_patches

def denoise_image(noisy_img, model, J=3, patch_size=9):
    # NSST decomposition
    low_freq, subbands = decompose(noisy_img, scales=J)
    H, W = noisy_img.shape
    pad = patch_size // 2
    
    # Process each subband
    denoised_bands = []
    for j in range(J):
        n_dir = subbands[j].shape[2]
        denoised_band = np.zeros_like(subbands[j])
        
        for d in range(n_dir):
            # Extract patches for current direction
            padded = np.pad(subbands[j][:, :, d], pad, mode='reflect')
            patches = view_as_windows(padded, (patch_size, patch_size))
            patches = patches.reshape(-1, patch_size, patch_size)
            
            # Predict class probabilities
            probs = model.predict(patches, verbose=0)
            edge_mask = (probs[:, 1] > 0.5).reshape(H, W)
            
            # Apply thresholding to non-edge coefficients
            band_data = subbands[j][:, :, d]
            sigma_hat = np.median(np.abs(band_data)) / 0.6745
            T_j = np.sqrt(2 * np.log(H*W)) * sigma_hat
            
            denoised = np.where(
                edge_mask,
                band_data,
                np.sign(band_data) * np.maximum(np.abs(band_data) - T_j, 0)
            )
            denoised_band[:, :, d] = denoised
        
        denoised_bands.append(denoised_band)
    
    return low_freq, denoised_bands
