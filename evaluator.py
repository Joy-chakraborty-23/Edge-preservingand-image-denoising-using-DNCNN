from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate(clean, denoised):
    psnr = peak_signal_noise_ratio(clean, denoised, data_range=255)
    ssim = structural_similarity(clean, denoised, data_range=255)
    return psnr, ssim
