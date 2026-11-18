# NumPy Implementation of OpenCLIP ViT-B-32 Image Encoder

A course project for Python Programming at Shenzhen University.

This project implements the forward pass of the OpenCLIP ViT-B-32 image encoder using NumPy.

## Overview

The encoder transforms preprocessed RGB images of shape `(B, 3, 224, 224)` into normalized image feature vectors of shape `(B, 512)`.

The implementation covers:

- Image patchification and patch projection
- CLS token and positional embeddings
- Multi-head self-attention
- Transformer blocks with LayerNorm, GELU, MLP, and residual connections
- CLS feature projection and L2 normalization

## Implementation

The core implementation is in `image_encoder.py`.

The encoder uses NumPy for forward computation. Model parameters are loaded from pretrained weights and are not trained or updated.

## Validation

The implementation was tested using the course-provided validation script.

All requested numerical checks passed:

| Stage | Output Shape | Maximum Absolute Error |
|---|---|---:|
| Preprocess | `(10, 3, 224, 224)` | `0` |
| Patches | `(2, 49, 3072)` | `0` |
| Tokens | `(2, 50, 768)` | `4.005e-05` |
| Block 0 | `(2, 50, 768)` | `7.063e-06` |
| Last tokens | `(2, 50, 768)` | `4.768e-05` |
| Output | `(10, 512)` | `5.439e-07` |
| Batch | `(3, 512)` | `4.619e-07` |

## Repository Contents

- `image_encoder.py`: NumPy image encoder implementation
- `helpers.py`: Numerical helper classes and functions
- `preprocess_image.py`: Image preprocessing
- `runtime.py`, `run.py`: Model loading and inference entry point
- `check.py`: Numerical validation script
- `config.json`: Model configuration
- `requirements.txt`: Python dependencies

## Data and Model Weights

Course-provided model weights (`weights/`) and reference assets (`assets/`) are intentionally excluded from this public repository because of their size and course-provided nature.

A fresh clone does not include all files required to run inference or the full numerical checks. To reproduce the experiments, obtain the required `weights/` and `assets/` directories from the original course materials and place them in the project root.

## Acknowledgements

This project was completed as part of a Python Programming course. ChatGPT was used as a learning and debugging assistant to explain concepts, review tensor shapes, and help investigate numerical issues.
