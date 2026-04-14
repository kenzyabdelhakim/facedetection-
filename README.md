# Skin Type Detection with Vision Transformer (ViT)

This project is a complete end-to-end image classification pipeline for skin type detection using a pretrained Vision Transformer (ViT) model from Hugging Face.

Dataset used: `dima806/skin-types-image-detection-vit` from Kaggle.

## Folder Structure

```text
image detection/
├── requirements.txt
├── README.md
└── src/
    ├── model.py
    ├── utils.py
    ├── train.py
    ├── inference.py
    └── embedded_inference.py
```

After training, output files are generated under:

```text
outputs/
├── raw_data/                         # Kaggle downloaded data
├── processed_data/
│   ├── train/
│   ├── val/
│   └── test/
├── artifacts/
│   ├── best_model.pth
│   ├── skin_vit_classifier_final.pth
│   ├── skin_vit_classifier.onnx
│   ├── skin_vit_classifier_int8.onnx
│   └── label_map.json
└── evaluation/
    ├── training_curves.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

## 1) Setup

### Create and activate virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

## 2) Configure Kaggle API

1. Go to Kaggle -> Account -> Create New API Token.
2. Download `kaggle.json`.
3. Place it at:
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
4. Ensure file permissions are secure.

Quick test:

```powershell
kaggle datasets list -s skin-types-image-detection-vit
```

## 3) Train the ViT Model

From project root:

```powershell
python src/train.py --epochs 10 --batch_size 16 --lr 3e-5
```

What this does:
- Downloads dataset from Kaggle
- Detects class folders automatically
- Creates train/val/test splits
- Applies preprocessing + augmentation
- Loads `google/vit-base-patch16-224` pretrained model
- Fine-tunes on your skin type dataset
- Prints training/validation loss and accuracy every epoch
- Evaluates on test set (accuracy, confusion matrix, classification report)
- Saves `.pth` model and exports ONNX + INT8 quantized ONNX

## 4) Inference on One Image (PyTorch)

```powershell
python src/inference.py --image "path\to\image.jpg"
```

## 5) Inference from Camera (PyTorch)

```powershell
python src/inference.py --use_camera --camera_id 0
```

- Press `SPACE` to capture frame and classify.
- Press `ESC` to cancel.

## 6) ONNX Inference (Single Image)

```powershell
python src/inference.py --onnx "outputs/artifacts/skin_vit_classifier_int8.onnx" --image "path\to\image.jpg"
```

## 7) Lightweight Embedded/Edge Inference

This script is designed for CPU-only edge devices (Raspberry Pi, mini-PC, external processor setups):

```powershell
python src/embedded_inference.py --model "outputs/artifacts/skin_vit_classifier_int8.onnx" --camera_id 0
```

It uses:
- `onnxruntime` (fast CPU runtime)
- OpenCV camera capture
- Quantized INT8 ONNX model for lower memory and better speed

## Notes for Arduino + External Processor

Arduino boards generally cannot run ViT directly due to memory/compute constraints. Typical deployment is:
- Arduino handles sensor/control logic
- External processor (Raspberry Pi / Jetson / mini Linux device) runs ONNX inference
- Communicate prediction result via serial/UART/I2C

## Useful Optional Training Arguments

```powershell
python src/train.py --epochs 20 --batch_size 8 --num_workers 4 --image_size 224
```

## Troubleshooting

- If Kaggle download fails:
  - Check `kaggle.json` location and credentials
  - Run `kaggle datasets list` to validate setup
- If CUDA is not available:
  - Training will run on CPU automatically (slower)
- If ONNX quantization fails:
  - Training still completes; only INT8 export is skipped

