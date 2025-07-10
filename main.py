import numpy as np
from .data_loader import load_ilsvrc2012_val
from .preprocessing import rgb2gray, add_gaussian_noise
from .edge_detector import canny_edges
from .nsst import decompose
from .patch_extractor import extract_patches
from .model import build_model
from .trainer import train_model
from .denoiser import denoise_image
from .evaluator import evaluate

def train_pipeline(data_dir, output_dir):
    # Step 1: Load and preprocess data
    clean_images = load_ilsvrc2012_val(data_dir)
    gray_images = rgb2gray(clean_images)
    
    # Step 2: Generate training data
    all_patches, all_labels = [], []
    for sigma in [10, 20, 30, 40, 50, 60, 70]:
        noisy_images = add_gaussian_noise(gray_images, sigma)
        edge_maps, directions = canny_edges(gray_images)
        
        for i in range(len(clean_images)):
            # NSST decomposition
            _, subbands = decompose(noisy_images[i])
            global_dir_idx = map_direction(directions[i])
            
            # Extract patches
            patches, labels = extract_patches(
                subbands, edge_maps[i], directions[i], global_dir_idx
            )
            all_patches.append(patches)
            all_labels.append(labels)
    
    X = np.vstack(all_patches)
    y = np.hstack(all_labels)
    
    # Step 3: Train model
    model = build_model()
    trained_model, history = train_model(model, X, y)
    trained_model.save(f"{output_dir}/denoising_model.h5")

def test_pipeline(test_images, model_path):
    model = tf.keras.models.load_model(model_path)
    results = []
    
    for img in test_images:
        noisy = add_gaussian_noise(img, 20)  # Example: σ=20
        low_freq, denoised_bands = denoise_image(noisy, model)
        denoised = reconstruct(low_freq, denoised_bands)
        psnr, ssim = evaluate(img, denoised)
        results.append((psnr, ssim))
    
    return results
