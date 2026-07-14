# UNet++ for Remote Sensing Change Detection

A PyTorch implementation of **UNet++ (Nested U-Net)** for binary change detection in very-high-resolution (VHR) satellite/aerial imagery.

## Overview

This project implements **UNet++** (Zhou et al., 2018, *"UNet++: A Nested U-Net Architecture for Medical Image Segmentation"*) applied to remote sensing change detection. The model takes a pair of co-registered images (pre-change and post-change) and outputs a pixel-wise binary change map.

The reference paper for the change detection dataset and context is: *Peng et al., "Remote Sensing Image Change Detection Based on a Siamese CNN and UNet++", Remote Sensing, 2019*.

## UNet vs UNet++

### Standard U-Net
- Encoder-decoder architecture with **plain skip connections**
- Features from encoder level `i` are directly concatenated to decoder level `i`
- Creates a semantic gap: shallow encoder features (high-res, low-semantics) are combined directly with deep decoder features (low-res, high-semantics)

### UNet++ (Nested U-Net)
- Replaces plain skip connections with **dense nested skip pathways**
- Each skip connection passes through a series of convolutional blocks (`X[i][j]` nodes)
- Each node `X[i][j]` receives concatenated outputs from **all previous nodes** at level `i` plus the upsampled output from level `i+1`
- This progressively bridges the semantic gap between encoder and decoder features
- Enables better gradient flow and multi-scale feature aggregation

### Architecture Comparison

```
U-Net:          UNet++:
                X[0][0] -> X[0][1] -> X[0][2] -> X[0][3] -> X[0][4]
                |          |          |          |
X[0] ---------- + --------- + --------- + --------- + ---> output
|               |          |          |          |
X[1] ---------- + --------- + --------- + ---> up
|               |          |          |
X[2] ---------- + --------- + ---> up
|               |          |
X[3] ---------- + ---> up
|
X[4] ---> up
```

## Model Architecture Details

### Dimensions at Each Level

| Level | Layer | Input Channels | Output Channels | Spatial Size (256x256 input) |
|-------|-------|---------------|-----------------|------------------------------|
| 0 | Encoder block 0 | 6 (3+3) | 32 | 256 x 256 |
| 1 | Encoder block 1 | 32 | 64 | 128 x 128 |
| 2 | Encoder block 2 | 64 | 128 | 64 x 64 |
| 3 | Encoder block 3 | 128 | 256 | 32 x 32 |
| 4 | Bottleneck | 256 | 512 | 16 x 16 |

### Nested Nodes (Decoder)

Each node at level `i` and depth `j` (denoted `X[i][j]`) receives:
- All `X[i][k]` for `k < j` (previous nodes at same level, concatenated)
- Upsampled `X[i+1][j-1]` (from deeper level)

Total nodes: 15 (including encoder nodes and nested decoder nodes)

### Parameters

- **Base filters**: 32 (doubles each level: 32, 64, 128, 256, 512)
- **Input**: 6 channels (RGB pre + RGB post concatenated)
- **Output**: 1 channel (binary change probability, sigmoid-activated)
- **Convolution**: 2x (Conv3x3 + BatchNorm + ReLU) per DoubleConv block
- **Upsampling**: Bilinear interpolation
- **Pooling**: 2x2 MaxPool

### Loss Function

**BCE + Dice Loss** (weighted combination):
```
Loss = 0.5 * BCELoss + 0.5 * DiceLoss
```

### Optimizer

- Adam with learning rate 1e-4

### Total Parameters

~9 million parameters. The model is **too large for most local GPUs** and requires a Colab GPU (T4 or P100) for training.

## Dataset

**Change Detection Dataset** - VHR satellite imagery pairs with binary change labels.

- **Source**: Google Drive ([download link](https://drive.usercontent.google.com/download?id=1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9&export=download&confirm=t&uuid=6ad01a3d-5864-4f0c-ae7e-642a14a3d197))
- **Size**: ~2500+ image pairs
- **Structure**:
  ```
  vhrdata/
    ChangeDetectionDataset/
      Real/
        subset/
          train/   {A/, B/, label/}
          val/     {A/, B/, label/}
  ```
- **Split**: Train/val split included
- **Resolution**: 256x256 (resized from original VHR)

## Project Structure

```
ChangeDetection-nnunet-paperimplementation/
  unetpp_change_detection.ipynb   # Main implementation notebook
  paper/
    notes.txt                     # Research notes
    remotesensing-11-01382-v2.pdf # Reference paper
  dump.txt                        # Useful links
  README.md
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- matplotlib
- numpy
- Pillow

Install with:

```bash
pip install torch torchvision matplotlib numpy Pillow
```

## Usage

1. Open `unetpp_change_detection.ipynb` in Google Colab (recommended) or Jupyter
2. Run cells sequentially to:
   - Download and extract the dataset
   - Build the UNet++ model
   - Train for 50 epochs
   - Evaluate with IoU and F1 metrics
   - Visualize predictions

## Expected Results

| Metric | Expected Range |
|--------|---------------|
| Validation IoU | 0.60 - 0.75 |
| Validation F1 | 0.75 - 0.85 |
| Validation Loss | 0.15 - 0.30 |

## References

- Zhou, Z., Siddiquee, M. M. R., Tajbakhsh, N., & Liang, J. (2018). UNet++: A nested U-net architecture for medical image segmentation. *MICCAI 2018*.
- Peng, D., Zhang, Y., & Guan, H. (2019). Remote sensing image change detection based on a Siamese CNN and UNet++. *Remote Sensing*, 11(11), 1382.