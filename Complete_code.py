import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import gaussian_laplace
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Dropout
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import tensorflow_datasets as tfds

# Dataset loader with resizing and normalization
def load_div2k(img_size=(128, 128), num_samples=None):
    dataset = tfds.load("div2k", split="train", as_supervised=True)
    images = []
    for img, _ in dataset.take(num_samples or -1):  # Take specified number of samples or all
        img_resized = tf.image.resize(img, img_size) / 255.0  # Normalize to [0, 1]
        images.append(img_resized.numpy())
    images = np.array(images, dtype=np.float32)
    print(f"Loaded {len(images)} images with shape: {images.shape}")
    return images

# Add random Gaussian noise
def add_gaussian_noise(images, sigma=0.05):
    noisy = images + np.random.normal(0, sigma, images.shape)
    noisy_clipped = np.clip(noisy, 0., 1.)
    return noisy_clipped.astype(np.float32)

# Edge-based loss component (Canny + Laplacian of Gaussian)
def edge_map(x):
    canny = feature.canny(x, sigma=1).astype(np.float32)
    log   = gaussian_laplace(x, sigma=1)
    return np.maximum(canny, (log > 0).astype(np.float32))

def edge_loss(y_true, y_pred):
    y_true_edges = tf.numpy_function(edge_map, [y_true], tf.float32)
    y_pred_edges = tf.numpy_function(edge_map, [y_pred], tf.float32)
    return tf.reduce_mean(tf.abs(y_true_edges - y_pred_edges))

# Residual block for a lightweight SRCNN-style model
def residual_block(x, filters, kernel_size=3, dropout_rate=0.0):
    skip = x
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([skip, x])
    x = Activation("relu")(x)
    return x

def build_sr_model(input_shape=(128, 128, 3), num_blocks=4, filters=64):
    inp = Input(shape=input_shape)
    x = inp
    for _ in range(num_blocks):
        x = residual_block(x, filters)
    out = Conv2D(3, 3, padding="same", activation="sigmoid")(x)
    return Model(inp, out, name="LightweightSR")

# Custom data generator
class SRDataGenerator(Sequence):
    def __init__(self, hr_images, batch_size=8, sigma=0.05):
        self.hr_images = hr_images
        self.batch_size = batch_size
        self.sigma = sigma
        self.indexes = np.arange(len(self.hr_images))

    def __len__(self):
        return int(np.ceil(len(self.hr_images) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        hr_batch = self.hr_images[batch_idx]
        lr_batch = add_gaussian_noise(hr_batch, self.sigma)
        return lr_batch, hr_batch

# Training pipeline
def train_sr_model(epochs=5, batch_size=8, num_samples=200):
    hr_images = load_div2k(num_samples=num_samples)
    gen = SRDataGenerator(hr_images, batch_size=batch_size)

    model = build_sr_model()
    model.compile(optimizer="adam",
                  loss=["mae", edge_loss],
                  loss_weights=[0.8, 0.2],
                  metrics=[psnr])

    history = model.fit(gen, epochs=epochs)
    return model, history

# Quick qualitative check
def show_results(model, num_images=3):
    hr_images = load_div2k(num_samples=num_images)
    lr_images = add_gaussian_noise(hr_images, sigma=0.05)
    sr_images = model.predict(lr_images)

    plt.figure(figsize=(12, 4 * num_images))
    for i in range(num_images):
        for j, (img, title) in enumerate(zip([lr_images[i], sr_images[i], hr_images[i]],
                                             ["Input (noisy)", "SR output", "Ground truth"])):

            plt.subplot(num_images, 3, 3 * i + j + 1)
            plt.imshow(img)
            plt.title(title)
            plt.axis("off")
    plt.tight_layout()
    plt.show()

# Run everything (demo)
if __name__ == "__main__":
    model, _ = train_sr_model(epochs=1, num_samples=40)  # quick demo run
    show_results(model, num_images=3)
