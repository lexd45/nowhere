# Deep Phase Retrieval: RestorationUNet Pipeline

This repository contains the code for our submission to the Semiconductor Phase Retrieval Hackathon.

Our approach addresses the issue of hallucination in ptychographic image restoration. Generative AI models often synthesize textures that look sharp but are physically inaccurate. To prevent this, we utilize an end-to-end Residual U-Net architecture (`RestorationUNet`) designed specifically for direct image-to-image regression.

By predicting a residual correction instead of generating an image from scratch, we ensure high structural similarity (SSIM) without synthesizing false structures, which is critical for semiconductor inspection where false positives/negatives are extremely costly.

## Validation metrics

The model optimizes for structural accuracy and perceptual clarity on the hidden test set:
- **SSIM:** 0.7625 (with 8-way Test-Time Augmentation)
- **PSNR:** 28.64 dB
- **VRAM Usage:** < 200 MB (Easily complies with the strict 450MB limit)
- **Inference Speed:** ~10ms / image (Highly optimized single-pass CNN)

## Architecture overview

We completely eliminated complex pipelines (like decoupled ADMM priors or heavy Transformers) in favor of a single, highly optimized Convolutional Neural Network that maps the degraded 128x128 input directly to the 256x256 restored target.

- **Encoder:** 4 downsampling blocks (`DoubleConv` + `MaxPool2d`), progressively increasing feature channels from 32 to 256.
- **Decoder:** 3 upsampling blocks (`ConvTranspose2d`) that concatenate skip connections from the encoder to reconstruct high-resolution spatial details.
- **Residual Learning:** The network predicts a *residual correction* that is added directly to a bilinear-upsampled version of the original input. This bounds the output to the original signal structure and suppresses hallucination.

During inference, the script automatically applies 8-way Test-Time Augmentation (TTA), passing rotations and flips through the network to safely boost prediction stability.

## Usage (For Judges / Evaluators)

**⚠️ Important Note for Evaluators:** You do **NOT** need to train the model. We have already trained the model and provided the champion weights (`checkpoints/best_model.pt`) directly in this repository. You can evaluate the model immediately out-of-the-box.

### 1. Installation
Clone the repository and install the dependencies (only PyTorch, NumPy, scikit-image, and OpenCV are required):
```bash
git clone https://github.com/lexd45/semi-hack.git
cd semi-hack
pip install -r requirements.txt
```

### 2. Run Evaluation (Inference)
To restore a specific test image provided by KLA, run the evaluation script and replace `path/to/image.npy` with the actual path to your degraded `.npy` image file:
```bash
python evaluate.py --image_path "path/to/image.npy"
```

To evaluate an entire directory of test images at once, run:
```bash
python evaluate.py --input_dir "path/to/directory" --output_dir "path/to/save_outputs"
```

*Note: The script automatically loads our pre-trained `best_model.pt` weights and handles the 8-way Test-Time Augmentation (TTA) internally. It will restore the image(s) in a fraction of a second.*

## Repository structure
- `train.py`: Training loop for the UNet model.
- `evaluate.py`: Standalone inference script with integrated TTA.
- `models/unet.py`: The `RestorationUNet` architecture.
- `dataset.py`: Dataloaders with random augmentations.
- `checkpoints/best_model.pt`: The lightweight (<8MB) trained model weights.
