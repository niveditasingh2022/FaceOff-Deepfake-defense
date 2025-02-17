# FaceOff-Deepfake-defense
FaceOff:  Safeguarding User Privacy with Real-Time Deepfake Defense

### Usage

1. Library Installation

   ```bash
   
   pip install opencv-python numpy pillow
   pip install insightface
   pip install onnxruntime-gpu==1.14.1

2. Faceswapper Model Download

   ```bash

   !wget https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx?download=true -O inswapper_128.onnx

3. FaceSwap

   ```bash

   python faceswap_video.py
