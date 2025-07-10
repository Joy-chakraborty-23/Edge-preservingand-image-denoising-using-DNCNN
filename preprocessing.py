import cv2
import numpy as np

def rgb2gray(images):
    return np.dot(images[..., :3], [0.299, 0.587, 0.114])

def add_gaussian_noise(images, sigma):
    noisy = images + np.random.normal(0, sigma, images.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)
