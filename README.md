# 👁️ Real-Time Obeject Detection with YOLO and OpenCV

This project demonstrates real-time face detection using a custom-trained **YOLO** model and a webcam, implemented in **Python** with **OpenCV**.

---

## 🚀 Features

- 🔍 Real-time detection using webcam feed  
- 📦 Powered by a YOLO model (`yolo11n.pt`)  
- 🎯 Bounding boxes with class name and confidence  
- 🧠 Lightweight and optimized for speed  

---

## 📂 Files

| File         | Description                            |
|--------------|----------------------------------------|
| `face.py`    | Main Python script for live detection  |
| `yolo11n.pt` | YOLO model weights *(place in `/Yolo-Wight`)*  
| `README.md`  | Project documentation (this file)  

---

## 🗂️ Project Structure

face-detection/
├── face.py
├── Yolo-Wight/
│   └── yolo11n.pt
├── README.md
└── [optional] output.jpg  # saved output image


## 🧰 Requirements

Install dependencies via pip:

```bash
pip install ultralytics opencv-python
