# 🧠 Object Detection using OpenCV and a Pretrained Model

This project demonstrates real-time object detection using OpenCV and a pre-trained **SSD MobileNet v3** model trained on the **COCO dataset**.

## 🚀 How It Works

The project uses the **TensorFlow Object Detection API** along with OpenCV's `dnn` module to perform object detection. Here’s how the system works:

1. **Model and Config Files**
   - `frozen_inference_graph.pb`: The pre-trained TensorFlow model containing the learned weights.
   - `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`: Configuration file that defines the model architecture for OpenCV.
   - `coco.names`: A text file that lists all 80 class names from the COCO dataset used for labeling detections.

2. **Preprocessing Input**
   - Input frames are converted into blobs using `cv2.dnn.blobFromImage()` before being passed to the neural network.

3. **Detection Logic**
   - The model returns bounding boxes, class IDs, and confidence scores.
   - Detections with confidence above a set threshold (e.g., 0.5) are visualized using OpenCV.

## 📁 Project Structure

| File/Folder                           | Description                                                                                            |
|---------------------------------------|--------------------------------------------------------------------------------------------------------|
| `main.py`                             | Main script that loads the model, performs detection, and visualizes results using OpenCV               |
| `coco.names`                          | List of class labels used by the COCO-trained model                                                     |
| `frozen_inference_graph.pb`           | Pre-trained model (frozen graph) used for inference                                                    |
| `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` | Configuration file for the SSD MobileNet v3 model                                                   |
| `lena.png`                            | Sample image for testing object detection                                                              |
| `venv/`                               | Python virtual environment with required packages installed                                           |
| `README.md`                           | Project documentation (this file)                                                                      |

## 🛠️ Requirements

Make sure to install the required packages:

```bash
pip install opencv-python
```


If using the provided virtual environment, activate it and install the packages from `requirements.txt`:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Running the Project

Run the object detection script with:

```bash
python main.py
```


You can modify the script to use a webcam feed or replace `lena.png` with another image for testing.

## 📦 Model Info

- **Model**: SSD MobileNet v3  
- **Trained On**: COCO dataset (80 classes)  
- **Framework**: TensorFlow  
- **Inference**: Performed using OpenCV’s `cv2.dnn` module for efficiency and portability

## 📽️ Demo

Watch the object detection in action: [YouTube Demo](https://www.youtube.com/watch?v=RgHvVXvja6I)
