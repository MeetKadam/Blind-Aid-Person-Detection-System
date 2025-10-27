
# 🧑‍🦯 Blind Aid Object Detection System

## Project Overview

This project implements a real-time object detection system designed to assist the visually impaired. It uses a **custom-built, simplified YOLO (You Only Look Once)** architecture trained from scratch using PyTorch to detect approaching hazards (specifically **Person** and **Car**) and provides auditory warnings via text-to-speech (TTS).

The primary goal was to create a functional deep learning pipeline *without* relying on high-level libraries (like Ultralytics or TorchVision pre-trained models) for the core training loop, loss function, or model architecture.

-----

## 🚀 Features

  * **Custom Architecture:** CNN backbone and detection head implemented entirely in `model.py`.
  * **Custom Training:** Implemented custom **YOLO Loss Function** in `train.py`.
  * **Real-Time Inference:** Decodes raw $7 \times 7 \times 7$ output tensor into usable bounding boxes.
  * **Robust Post-Processing:** Includes custom **Non-Maximum Suppression (NMS)** logic to clean up multiple bounding box predictions.
  * **Temporal Smoothing:** Uses a distance history buffer to stabilize noisy distance readings.
  * **Proximity Alert:** Issues a voice warning ("Warning, 1 person approaching...") based on object size and estimated distance.

-----

## 🛠️ Setup and Installation

### Prerequisites

You need Python 3.8+ and access to your command line/terminal.

1.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```
2.  **Install Dependencies:**
    ```bash
    pip install torch torchvision numpy opencv-python pyttsx3
    ```

### Project Structure

Ensure your project directory contains the following files and folders:

```
BlindAid-Detector/
├── model.py            # Custom CNN Architecture definition
├── utils.py            # Data loading, YOLO-to-Grid conversion, and IoU function
├── detect_person_main.py   # REAL-TIME INFERENCE & ALERT SCRIPT (The main program)
├── basic_detector_model.pth # TRAINED WEIGHTS FILE (The output of Person_Detector_Model_Training.ipynb)
└── README.md
```

-----

## 📦 Usage

### Phase 1: Training (Already Completed)

The training process was completed using a custom $7 \times 7$ grid, $224 \times 224$ image size, and a custom PyTorch training loop to generate the required weights.

**Resulting File:** `basic_detector_model.pth`

### Phase 2: Real-Time Detection and Alert

This phase uses the trained weights (`.pth` file) and your mobile camera feed to run the live assistant.

#### A. Set up Mobile Camera Feed

1.  **Install IP Webcam App** (e.g., "IP Webcam" for Android) on your smartphone.

2.  **Connect Both Devices** to the same Wi-Fi network.

3.  **Start the server** in the app and note the stream URL (e.g., `http://192.168.1.5:8080`).

4.  **Edit `detect_person2.py`:** Update the camera source line:

    ```python
    # Update this line with your phone's stream URL
    cap = cv2.VideoCapture('http://<YOUR_PHONE_IP_ADDRESS>:8080/video')
    ```

#### B. Run the Assistant

Execute the main inference script from your terminal:

```bash
python detect_person2.py
```

-----

## ⚙️ Key Configuration Parameters

The performance of the real-time system is entirely controlled by the thresholds in `detect_person2.py`:

| Variable | Recommended Adjustment | Purpose |
| :--- | :--- | :--- |
| `CONFIDENCE_THRESHOLD` (e.g., 0.45) | **Lower to $\mathbf{0.35}$** if missing objects. **Raise to $\mathbf{0.6}$** if detecting too many false alarms. | Minimum confidence score to accept a detection. |
| `NMS_IOU_THRESHOLD` (e.g., 0.01) | **Raise to $\mathbf{0.50}$** if boxes are jittering/overlapping too much (overly aggressive NMS). | IoU threshold for removing duplicate bounding boxes. |
| `KNOWN_WIDTH` (e.g., 180) | **Lower this value** to reduce the estimated distance (make objects seem closer). **Raise this value** to increase the estimated distance. | Calibration constant for distance estimation (crucial for accurate alerts). |
| `DISTANCE_BUFFER_SIZE` (e.g., 4) | **Increase** to $\mathbf{7}$ or $\mathbf{10}$ for smoother distance readings. | Number of recent frames used to calculate the median distance. |

-----

## Project Conclusion

This project successfully demonstrates the implementation of a full object detection workflow using fundamental deep learning techniques in PyTorch, resulting in a functional, personalized safety assistant prototype.
