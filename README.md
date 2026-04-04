
---

# Resource-Efficient Adaptive Keyframe Selection for Scalable 3D Gaussian Splatting

[![Journal: The Visual Computer](https://img.shields.io/badge/Journal-The%20Visual%20Computer-blue)](https://www.springer.com/journal/371)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19414672.svg)](https://doi.org/10.5281/zenodo.19414672)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)

> **Notice**: This repository provides the official implementation of **REAKS**, a framework directly related to the manuscript submitted to *The Visual Computer*. Users are encouraged to cite the relevant work as outlined in the [Citation](#-citation) section.

---

## 📝 Abstract

3D Gaussian Splatting (3DGS) has emerged as a highly efficient neural rendering technique for high-fidelity 3D reconstruction. However, its practical deployment is severely hindered by excessive memory consumption and computational overload in large-scale scene processing, leading to out-of-memory (OOM) errors and limited scalability. 

This repository presents **REAKS**, a spectral clustering-driven adaptive keyframe selection framework. By integrating multi-source hierarchical feature fusion and dynamic quantile-based similarity matrix construction, REAKS achieves view-adaptive spectral clustering with automated cluster number optimization. Experimental results on natural and medical endoscopic datasets demonstrate that REAKS accelerates Structure-from-Motion (SfM) by approximately **12x** and reduces peak GPU memory usage by over **37.5%** while maintaining high-fidelity rendering accuracy.

---

## 🛠️ Reproducibility & Implementation Details

To ensure the transparency and reproducibility required by the scientific community, we provide the following technical guidelines.

### 1. Requirements & Dependencies
- **OS**: Ubuntu 20.04 LTS
- **Hardware**: NVIDIA GPU (RTX 3090 or higher recommended for 24GB VRAM capability)
- **Software**: 
  - CUDA 11.8+
  - Python 3.8+
  - PyTorch 1.13+
  - Scikit-learn (for Spectral Clustering)

### 2. Core Algorithm: `REAKS.py`
The primary logic for adaptive keyframe selection is encapsulated in `REAKS.py`. It implements:
- **Feature Extraction**: Multi-source hierarchical fusion for image descriptors.
- **Spectral Clustering**: View-adaptive grouping of redundant frames.
- **Adaptive Selection**: Automated optimization of cluster counts to balance spatial distribution and redundancy.

### 3. Usage Instructions
To replicate the efficiency benchmarks reported in our paper:

1. **Pre-processing (Keyframe Selection)**:
   ```bash
   python REAKS.py --input_dir <path_to_raw_images> --output_dir <path_to_selected_keyframes> --threshold 0.85
   ```
2. **Reconstruction (3DGS Training)**:
   Use the selected keyframes as input for the 3DGS training pipeline:
   ```bash
   python train.py -s <path to COLMAP or NeRF Synthetic dataset>
   ```

---

## 📊 Empirical Validation
| Framework | SfM Speedup | Peak Memory Reduction | Rendering Quality (PSNR) |
| :--- | :--- | :--- | :--- |
| Original 3DGS | 1.0x | 0% | Baseline |
| **REAKS (Ours)** | **~12.0x** | **> 37.5%** | **Comparable** |

---

## 📂 Citation
If you find this work useful for your research, please cite our manuscript:

```bibtex
@article{zhang2026reaks,
  title={Resource-Efficient Adaptive Keyframe Selection for Scalable 3D Gaussian Splatting},
  author={Zhang, Peizhao and others},
  journal={The Visual Computer (Submitted)},
  year={2026},
  note={Directly related to the open-source implementation at https://github.com/PeiZhaoZhang/REAKS}
}
```

## 🙏 Acknowledgements
This codebase is developed as an optimization module for the [Official 3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) implementation. We acknowledge the foundational work of the 3DGS authors in the field of computer graphics.

---