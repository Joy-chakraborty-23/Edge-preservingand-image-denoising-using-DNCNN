import numpy as np
from skimage.util import view_as_windows

def extract_patches(subbands, edge_map, directions, global_direction_idx, patch_size=9):
    J = len(subbands)
    H, W = edge_map.shape
    patches, labels = [], []
    
    # Pad subbands
    pad = patch_size // 2
    padded_bands = [np.pad(b, [(pad, pad), (pad, pad), (0, 0)], mode='reflect') 
                    for b in subbands]
    
    for y in range(H):
        for x in range(W):
            # Get direction index for this pixel
            d = global_direction_idx[y, x]
            
            # Extract multiscale patch
            patch_stack = []
            for j in range(J):
                n_dir = padded_bands[j].shape[2]
                d_j = int(d * n_dir / 8)  # Map to current scale
                band = padded_bands[j][y:y+patch_size, x:x+patch_size, d_j]
                patch_stack.append(band)
            
            # Stack along depth (J channels)
            patch_3d = np.dstack(patch_stack)
            patches.append(patch_3d)
            labels.append(1 if edge_map[y, x] > 0 else 0)
    
    return np.array(patches), np.array(labels)
