# Mask CoMER: Enhancing Handwritten Mathematical Expression Recognition with Masked Language Pretraining and Regularization (Coming Soon!)

<p align="center">
  <a href="#"><img alt="Paper" src="https://img.shields.io/badge/Paper-PDF-blue?logo=readthedocs"></a>
  &nbsp; | &nbsp;
  <img alt="State of the Art" src="https://img.shields.io/badge/SOTA-Yes-success">
  &nbsp; | &nbsp;
  <img alt="License" src="https://img.shields.io/badge/License-TBD-lightgrey">
</p>

---

## 📝 Abstract

> Handwritten Mathematical Expression Recognition (HMER) is crucial for enhancing human-machine interaction in domains such as digitized education and automated document processing. While existing transformer-based methods employ coverage mechanisms to capture context, they often still lose critical cues that a human could quickly correct. To address this, we propose Mask CoMER, a two-stage training approach that combines a novel Masked Language Model pretraining objective with stochastic depth regularization. This method enhances contextual understanding and mitigates information loss, leading to improved recognition accuracy. Experiments on the CROHME 2014, 2016, and 2019 datasets demonstrate that Mask CoMER achieves state-of-the-art ExpRates of 64.56\%, 63.03\%, and 65.22\%, respectively. These results outperform the baseline CoMER by up to 5\% and surpass the previous SOTA by nearly 2\%

---

## ⚙️ Installation

### 1. Create environment
```bash
conda create -n maskcomer python=3.11 -y
conda activate maskcomer
```

### 2. Install PyTorch

Make sure to install the PyTorch version compatible with your CUDA toolkit.
For example, with CUDA 11.1:

```bash
pip install torch==1.8.1 torchvision==0.2.2 cudatoolkit==11.1
```

### 3. Install additional dependencies

```bash
pip install -r requirements.txt
```

---

## 📚 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{nampvh,
  title={Mask CoMER: Enhancing Handwritten Mathematical Expression Recognition with Masked Language Pretraining and Regularization},
  year={2025}
}
