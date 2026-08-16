# Deep Phase Retrieval: ADMM-NAFNet Pipeline

This repository contains the code for our submission to the Semiconductor Phase Retrieval Hackathon.

Our approach addresses the issue of hallucination in ptychographic image restoration. Generative AI models often synthesize textures that look sharp but are physically inaccurate. To prevent this, we anchor our neural networks to the physical data by combining the Alternating Direction Method of Multipliers (ADMM) with a NAFNet architecture.

This ensures high structural similarity (SSIM) without synthesizing false structures, which is necessary for semiconductor inspection.

## Validation metrics

The model optimizes for structural accuracy and perceptual clarity:
- **SSIM:** 0.8010 (Final validation with TTOPI patch blending)
- **PSNR:** 29.21 dB
- **LPIPS:** 0.2052
- **VRAM Usage:** < 450 MB (Fits easily within the H100 memory limits)
- **Inference Speed:** ~0.75s / image

## Architecture overview

The pipeline operates in three stages:

### Stage 1: ADMM physics prior
We first process the raw sensor data using ADMM. This step provides a sparse physical prior, giving the downstream neural network a structural baseline that prevents hallucinated defects.

### Stage 2: NAFNet base denoising
The ADMM output is passed through a NAFNet (Nonlinear Activation Free Network) model to correct global illumination, enforce structural continuity, and remove macro-level noise.

### Stage 3: Deep fusion and patch refinement
To recover high-frequency details, we use a deep fusion network with a `PixelShuffle(2)` layer that dynamically upscales the physical edges. 

During inference, we use Test-Time Overlapping Patch Inference (TTOPI). The images are split into overlapping 128x128 patches and blended using a 2D Gaussian window. This limits the network's field of view, forcing it to reconstruct local, high-frequency physical geometry instead of general background structures.

## Usage

### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/deep-phase-retrieval.git
cd deep-phase-retrieval
pip install -r requirements.txt
```

### Inference
To run the evaluation script on the test dataset:
```bash
python evaluation.py --checkpoint checkpoints_fusion_deep/best_deep_ema.pt --data_dir data/test_set
```
This script applies 4-way test-time augmentation (horizontal and vertical flips) and patch blending.

### Training
To train the deep fusion model:
```bash
python train.py
```
The training script uses a combined loss function (SSIM, L1, Gradient, and FFT loss) and applies gradient clipping to stabilize the network.

## Repository structure
- `train.py`: Training loop for the fusion network.
- `evaluation.py`: TTOPI inference script.
- `models/`: NAFNet and ADMM block architectures.
- `dataset.py`: Dataloaders for multi-modal fusion pairing.
