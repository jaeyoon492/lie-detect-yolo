# YOLOv8 Custom Model - [MapleStory Lie Detector Detector]

## Introduction

This project is a custom YOLOv8 model trained to detect [MapleStory Lie Detector Image Detection]. The model has been trained using a dataset containing [1903 Lie Detector Images] and is capable of accurately identifying [the clickable lie detector] in images.

## Model Details

- **Model Type**: YOLOv8
- **Training Dataset**: [https://universe.roboflow.com/andy-de-gheldere-fmqy2/ldldldld/dataset/2]
- **Number of Classes**: [6]
- **Classes**: 
  - 0: ['0']
  - 1: ['1']
  - 2: ['captcha']
  - 3: ['color']
  - 4: ['origin']
  - 5: ['star']
- **Training Epochs**: [100]
- **Image Size**: [640x640]
- **Learning Rate**: [Default]
- **Best Model**: [runs/detect/train8/weights/best.pt]

## Installation

To use this project, you need to have Python and the required dependencies installed. You can follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/jaeyoon492/lie-detect-yolo.git
    cd lie-detect-yolo
    ```

2. **Install Dependencies**:
    Install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

    Alternatively, you can install the necessary packages manually:
    ```bash
    pip install ultralytics opencv-python matplotlib
    ```


## Usage

### 1. Running Inference

To run inference using the trained model, use the following script:

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model = YOLO("path/to/your/trained_model.pt")

# Load and preprocess the image
image_path = "path/to/your/image.jpg"
image = cv2.imread(image_path)

# Run inference
confidence_threshold = 0.5
results = model.predict(source=image, imgsz=640, conf=confidence_threshold)

# Display the results
results.show()
```

## Troubleshooting CUDA Issues

If you encounter issues related to CUDA while running the model, such as errors related to CUDA backend or the model not utilizing the GPU, it might be due to an incorrect installation of PyTorch and its dependencies. Here’s how you can troubleshoot and resolve these issues.

### 1. Check PyTorch and CUDA Compatibility

First, verify that your PyTorch installation is configured to use CUDA:

```python
import torch

print("CUDA available:", torch.cuda.is_available())
```

### 2. Installing the Correct Version of PyTorch

Uninstall the current PyTorch version:
```bash
pip uninstall torch torchvision torchaudio
```

Reinstall PyTorch with the appropriate CUDA version:

You can install the correct version of PyTorch by specifying the CUDA version. For example, if you have CUDA 12.5 installed, use the following command:

```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Verifying the Installation

```python
import torch

print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU available:", torch.cuda.device_count())
```

### 4. Common Issues and Solutions
CUDA Not Found Error: Ensure that the CUDA toolkit and drivers are properly installed on your system. You can check your CUDA installation by running:
```bash
nvcc --version
```

### 5. Fallback to CPU
If you are unable to resolve CUDA-related issues, or if you are working in an environment without GPU access, you can modify the code to run on the CPU:

```python
device = torch.device('cpu')
model = YOLO("path/to/your/trained_model.pt", device=device)

# Run inference on the CPU
results = model.predict(source=image, imgsz=640, conf=0.5)
