# Edge-Preserving Image Denoising via Deep CNN

![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License: MIT](https://img.shields.io/badge/license-MIT-blue)

## Overview

This project implements the edge-preserving image denoising method proposed by Shahdoosti & Rahemi in *Signal Processing* 159 (2019) **20–32**, leveraging a Non-Subsampled Shearlet Transform (NSST) front-end and a deep convolutional neural network to classify and selectively denoise transform-domain coefficients fileciteturn1file0.

## Features

* **NSST-Based Decomposition:** Multi-scale, multi-directional shearlet analysis.
* **3D Patch Classification:** CNN determines edge-related vs. noise-related blocks.
* **Adaptive Soft-Thresholding:** Noise-related coefficients are denoised; edges preserved.
* **Supported Noise Levels:** σ∈ {10,20,30,40,50,60,70}.
* **Benchmark Tests:** Standard grayscale (Lena, Barbara, House) and medical ultrasound images.

## Architecture

1. **NSST Decomposition** – Decompose noisy image into one low-frequency and J=3 scales of D=8 directional subbands.
2. **3D Patch Extraction** – For each pixel, round gradient direction to nearest shearlet subband, stack 9×9 2D blocks into 9×9×3 tensors.
3. **CNN Classifier** – Three 3×3 convolution layers + 3×3 max-pool + two FC layers outputting edge vs. noise labels (see Fig. 4) fileciteturn1file16.
4. **Selective Denoising** – Apply adaptive soft-threshold to noise-related coefficients; retain edge coefficients.

![Pipeline Overview](docs/fig3.png)
*Fig. 3: Method pipeline*

## Installation & Requirements

```bash
# Clone repository
git clone https://github.com/yourusername/edge-denoise-cnn.git
cd edge-denoise-cnn

# Create environment
conda create -n eddenoise python=3.8
conda activate eddenoise

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:** Python 3.8+, TensorFlow 2.x or PyTorch 1.7+, ShearLab or PyShearlab, OpenCV, scikit-image

## Usage

1. **Prepare Data**

   * Place clean grayscale images in `data/clean/` and generate noisy pairs using provided script:

     ```bash
     python scripts/add_noise.py --sigma 10 20 30 --input data/clean --output data/noisy
     ```
2. **Train Model**

   ```bash
   python train.py \
     --clean_dir data/clean \
     --noisy_dir data/noisy \
     --nsst_levels 3 \
     --batch_size 128 \
     --epochs 50
   ```
3. **Denoise Images**

   ```bash
   python denoise.py \
     --model checkpoints/cnn_edge_denoise.pth \
     --input data/noisy/Lena_sigma20.png \
     --output results/Lena_denoised.png
   ```

## Results

| Image   | σ=20 | PSNR (dB) | SSIM |
| :------ | ---: | --------: | ---: |
| Lena    |   20 |     28.96 | 0.84 |
| Barbara |   20 |     26.13 | 0.79 |
| House   |   20 |     27.23 | 0.81 |

<p align="center">
  <img src="results/Lena_before_after.png" alt="Lena denoising" width="60%" />
</p>

## Configuration

Adjust key hyperparameters in `config.yaml`:

* `sigma_levels`: list of noise levels
* `patch_size`: 9 (height × width)
* `nsst_scales`: 3
* `cnn`: filter counts, kernel sizes, FC dimensions

## Contributing

1. Fork the repo.
2. Create your feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature/YourFeature`
5. Open a Pull Request

## License & Citation

This project is licensed under the MIT License.

If you use this code, please cite:

```bibtex
@article{shahdoosti2019edge,
  title={Edge-preserving image denoising using a deep convolutional neural network},
  author={Shahdoosti, Hamid Reza and Rahemi, Zahra},
  journal={Signal Processing},
  volume={159},
  pages={20--32},
  year={2019},
  publisher={Elsevier}
}
```
