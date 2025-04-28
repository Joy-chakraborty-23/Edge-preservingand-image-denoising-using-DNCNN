# Edge-preservingand-image-denoising-using-DNCNN

Below is a revised **README.md** with the neural-network diagram and its download link removed. Everything else is unchanged.

```markdown
# Lightweight Edge-Aware Image Super-Resolution  
*A residual CNN baseline with DIV2K, Canny + LoG edge loss, and minimal dependencies*

<div align="center">
  <img src="https://img.shields.io/badge/TF-2.x-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Python-3.9%2B-yellow?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square"/>
</div>

---

## ✨ Overview
This repository contains a concise yet powerful super-resolution (SR) model aimed at **research prototypes, education, and quick benchmarks**.

* **Residual CNN** (4 lightweight blocks, ~300 k params).  
* **Edge-aware training** – combines _L1_ with a Canny + LoG edge loss to boost perceptual sharpness.  
* **Noise robustness** – trains on Gaussian-noisy inputs so the network learns both de-noising and SR.  
* **Self-contained** – one file, no custom C/CUDA ops.  
* **TensorFlow Datasets** – pulls DIV2K automatically.

<p align="center">
  <img alt="Qualitative result" src="docs/example_grid.png" width="600">
  <br>
  <sup>Left → noisy input &nbsp;&nbsp;|&nbsp;&nbsp; center → network output &nbsp;&nbsp;|&nbsp;&nbsp; right → clean ground-truth</sup>
</p>

---

## 🧩 Network Architecture

| Stage | Details | Output size |
|-------|---------|-------------|
| **Input** | RGB patch | 128 × 128 × 3 |
| **Residual Block × 4** | Conv(3 × 3, 64) → BN → ReLU → _(optional Dropout)_ → Conv(3 × 3, 64) → BN → Add skip → ReLU | 128 × 128 × 64 |
| **Final Conv** | Conv(3 × 3, 3) + Sigmoid | 128 × 128 × 3 |
| **Output** | Super-resolved patch | 128 × 128 × 3 |

<details>
<summary>Why so small?</summary>

A four-block backbone is enough to demonstrate the **edge loss** and training loop without eating GPU memory or requiring long runtimes.  
Swap in more blocks or a deeper UNet if you need higher PSNR.
</details>

---

## 🔧 Installation

```bash
git clone https://github.com/your-username/lightweight-sr.git
cd lightweight-sr
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Dependencies**

| Package | Tested version |
|---------|----------------|
| `tensorflow` | 2.17 |
| `tensorflow-datasets` | 4.9 |
| `numpy` · `matplotlib` · `scikit-image` · `scipy` | latest |

> **Tip** – Installing TensorFlow with GPU support is strongly recommended for >10 k iters.

---

## 🚀 Quickstart

Train a toy model on **40** DIV2K images for **1 epoch**:

```bash
python sr_edge.py --epochs 1 --num_samples 40 --batch_size 8
```

That finishes in a couple of minutes on a single GPU.  
After training, a qualitative demo window pops up automatically.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `5` | Training epochs |
| `--batch_size` | `8` | Mini-batch size |
| `--num_samples` | `200` | Number of DIV2K crops to load |
| `--sigma` | `0.05` | Gaussian σ for synthetic noise |

---

## 📚 Code Walkthrough

```
sr_edge.py
│
├── load_div2k()      # tfds + resize + [0,1] float32
├── add_gaussian_noise()
│
├── edge_map()        # Canny + Laplacian-of-Gaussian
├── edge_loss()       # |edge(gt) − edge(pred)|
│
├── residual_block()  # Conv-BN-ReLU-(Dropout)-Conv-BN + skip
├── build_sr_model()  # 4 residual blocks + final conv
│
├── SRDataGenerator   # tf.keras.utils.Sequence wrapper
├── train_sr_model()  # compile(), fit()
└── show_results()    # matplotlib grid
```

---

## 📝 Loss Function Details

| Component | Expression | Weight |
|-----------|------------|--------|
| **Pixel L1** | `MAE(y_true, y_pred)` | **0.8** |
| **Edge** | `mean(|edge(y_true) − edge(y_pred)|)` | **0.2** |

Where  

```
edge(x) = max( Canny(x,σ=1),
               1{ LoG(x,σ=1) > 0 } )
```

The edge term encourages crisp structures that pure *L1* often blurs.

---

## 🔍 Evaluation

Run **PSNR** on the held-out validation set:

```python
from sr_edge import load_div2k, add_gaussian_noise, psnr

# Load val images
val = load_div2k(num_samples=100)
lr  = add_gaussian_noise(val)

psnr_vals = psnr(val, model.predict(lr))
print("Mean PSNR:", psnr_vals.numpy())
```

Add SSIM, LPIPS, or your favourite perceptual metric as needed.

---

## 📂 Project Structure

```
.
├── sr_edge.py
├── README.md
├── requirements.txt
└── docs/
    └── example_grid.png
```

> Keep heavy sample outputs under `docs/` so they render in the GitHub UI but stay out of your Python package path.

---

## 🗺️ Roadmap / TODO

- [ ] Replace Gaussian noise with authentic low-resolution down-sampling.  
- [ ] WandB / TensorBoard integration.  
- [ ] Quantize for mobile deployment (TFLite).  
- [ ] Add UNet-SRGAN variant.

---

## 📜 License

This project is released under the **MIT License** – see `LICENSE` for details.  
DIV2K is distributed by the original authors under a separate **Creative Commons BY-NC-SA 4.0** license.

---

## 🙏 Acknowledgements

* [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset  
* Kaiming He *et al.* – **“Deep Residual Learning for Image Recognition”**  
* skimage team for high-quality image processing utilities.

---

## ✍️ Citation

If you build on this repo, feel free to cite it as:

```text
@misc{lightweightSR2025,
  author       = {Your Name},
  title        = {Lightweight Edge-Aware Image Super-Resolution},
  year         = 2025,
  howpublished = {\url{https://github.com/your-username/lightweight-sr}}
}
```

Happy super-resolving! 🚀
```

---
