# Enhancing Distilled Datasets via Natural Data Mixing

**Authors:** Ian Pons, Guilherme B. Stern, Anna H. Reali Costa, Artur Jordao  
**Affiliation:** Escola Politécnica, Universidade de São Paulo (Poli-USP)

## Abstract
Dataset distillation emerges as a promising technique to reduce web-scale datasets into a compact version with only a few samples per class. However, existing techniques fail to fully capture the underlying properties of original (natural) training samples, leading to a generalization gap. 

In this work, we propose a simple yet effective mechanism to enhance distilled images. Our method transfers powerful and discriminative characteristics from natural images to distilled samples through a simple mixing process.

## Experimental Results
Our method consistently improves generalization across various benchmarks (CIFAR-10, CIFAR-100, SVHN, ImageNet Subsets).

### Performance on CIFAR-10 (Accuracy %)

| Method (IPC=50) | Baseline | **Baseline + Ours** | Gain (p.p.) |
| :--- | :---: | :---: | :---: |
| **DC** | 54.65 | **62.88** | +8.23 |
| **DM** | 58.43 | **71.61** | +13.18 |
| **DSA** | 53.32 | **71.10** | +17.78 |
| **ATT** | 70.00 | **76.00** | +6.00 |

> *Table 1 from the paper: Comparison across different distillation methods.*

## Citation
If you find this code or our paper useful for your research, please consider citing
```
@inproceedings{pons2025enhancing,
  title={Enhancing Distilled Datasets via Natural Data Mixing},
  author={Pons, Ian and Stern, Guilherme B. and Costa, Anna H. Reali and Jordao, Artur},
  booktitle={Conference on Graphics, Patterns and Images (SIBRAPI)},
  year={2025}
}
```
