import torch
import torch.nn.functional as F
import numpy as np
import cv2
# pyttsx3 is REMOVED
import time
import os
from torchvision import transforms

# --- 1. IMPORT CUSTOM MODULES ---
from model import BasicYOLODetector 
from utils import intersection_over_union

# --- 2. CONFIGURATION (MUST MATCH TRAINING) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE_S = 7
BOXES_PER_CELL_B = 1
NUM_CLASSES = 2 
IMG_SIZE = 224 # CRITICAL: MUST MATCH YOUR TRAINED MODEL

# --- 3. PROTOTYPE THRESHOLDS (ADJUSTED FOR STABILITY AND SENSITIVITY) ---
CONFIDENCE_THRESHOLD = 0.45      # Lowered to 0.40 to capture weaker detections
NMS_IOU_THRESHOLD = 0.01         # Balanced value for NMS
APPROACHING_THRESHOLD = 0.04     # Area proportion for alerting
DISTANCE_BUFFER_SIZE = 4         # Number of frames to average the distance over
ALERT_COOLDOWN = 7 
DISTANCE_CHANGE_THRESHOLD = 0.4 

# --- 4. CALIBRATION CONSTANTS ---
KNOWN_DISTANCE = 1.0  
KNOWN_WIDTH = 180     

CLASS_NAMES = ["person", "car"] 

# --- 5. TEXT-TO-SPEECH & DISTANCE HELPERS ---

def speak(text):
    """REMOVED TTS FUNCTIONALITY - ONLY PRINTS TO CONSOLE."""
    print(f"🔊 [ALERT]: {text}")

def estimate_distance(box_width_pixels):
    """Estimates distance using the standard pinhole model."""
    if box_width_pixels == 0:
        return float('inf')
    return (KNOWN_WIDTH * KNOWN_DISTANCE) / box_width_pixels

# --- 6. THE CORE DECODING LOGIC (UNCHANGED) ---

def decode_predictions(predictions, threshold):
    S, C = GRID_SIZE_S, NUM_CLASSES
    all_bboxes = []
    
    for i in range(S):
        for j in range(S):
            cell_pred = predictions[i, j, :]
            confidence = cell_pred[4].item()
            
            if confidence > threshold:
                x_center_cell = cell_pred[0].item()
                y_center_cell = cell_pred[1].item()
                width = cell_pred[2].item()
                height = cell_pred[3].item()
                
                class_probs = cell_pred[5:].float()
                class_score, class_id = torch.max(class_probs, dim=0)
                class_id = class_id.item()
                
                x_center = (x_center_cell + j) / S
                y_center = (y_center_cell + i) / S
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                all_bboxes.append([
                    class_id, 
                    confidence * class_score.item(),
                    x1, y1, x2, y2
                ])
                
    return all_bboxes

# --- 7. NON-MAXIMUM SUPPRESSION (UNCHANGED) ---

def non_max_suppression(bboxes, iou_threshold, conf_threshold, box_format="corners"):
    if len(bboxes) == 0:
        return []

    # Sort by confidence
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes_after_nms.append(chosen_box)
        
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]).unsqueeze(0),
                torch.tensor(box[2:]).unsqueeze(0),
                box_format="corners"
            ).item() < iou_threshold
        ]

    return bboxes_after_nms


# --- 8. MAIN EXECUTION ---

def main_inference():
    global LAST_SPOKEN_STATUS # Reference the global variable for state management

    custom_model = BasicYOLODetector(S=GRID_SIZE_S, B=BOXES_PER_CELL_B, C=NUM_CLASSES).to(DEVICE)
    model_path = "basic_detector_model.pth"
    
    try:
        custom_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        custom_model.eval()
    except FileNotFoundError:
        print(f"FATAL ERROR: Model weights '{model_path}' not found. Did you download them?")
        return
    
    cap = cv2.VideoCapture('http://192.168.0.100:8080/video') 

    if not cap.isOpened():
        print("Error: Could not open video stream. Check IP address or network connection.")
        return

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    # State Management Variables
    last_alert_time = 0
    last_spoken_distance = float('inf') 
    last_spoken_count = 0 
    LAST_SPOKEN_STATUS = None
    distance_history = [] 

    speak("System activated. Detecting objects.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()
        
        # --- A. Inference Pipeline ---
        input_tensor = transform(frame).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            raw_predictions = custom_model(input_tensor)
        
        raw_predictions = raw_predictions.squeeze(0).cpu() 
        
        # NOTE: decode_predictions threshold is passed here
        decoded_boxes = decode_predictions(raw_predictions, threshold=CONFIDENCE_THRESHOLD)
        final_boxes = non_max_suppression(decoded_boxes, iou_threshold=NMS_IOU_THRESHOLD, conf_threshold=CONFIDENCE_THRESHOLD)
        
        person_detections = [box for box in final_boxes if CLASS_NAMES[box[0]] == "person"]

        approaching_people_distances = []
        frame_h, frame_w, _ = frame.shape
        display_frame = frame.copy()

        # --- B. Distance and Visualization ---
        for box in person_detections:
            class_id, confidence, x1, y1, x2, y2 = box
            
            # Rescale normalized coordinates (0-1) back to pixel values
            x1_pix = int(x1 * frame_w)
            y1_pix = int(y1 * frame_h)
            x2_pix = int(x2 * frame_w)
            y2_pix = int(y2 * frame_h)

            box_w_pix = x2_pix - x1_pix
            box_h_pix = y2_pix - y1_pix

            normalized_area = (box_w_pix * box_h_pix) / (frame_w * frame_h)
            
            cv2.rectangle(display_frame, (x1_pix, y1_pix), (x2_pix, y2_pix), (0, 255, 0), 2)
            label = f"{CLASS_NAMES[class_id]}: {confidence:.2f}"
            
            # --- Display Distance on Screen (Optional, for debugging) ---
            if normalized_area > APPROACHING_THRESHOLD:
                distance = estimate_distance(box_w_pix)
                approaching_people_distances.append(distance)
                
                # Update visual label with distance for debugging
                label += f" | D:{distance:.1f}m"

            cv2.putText(display_frame, label, (x1_pix, y1_pix - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- C. Temporal Smoothing ---
        
        if approaching_people_distances:
            current_closest_distance = min(approaching_people_distances)
            
            distance_history.append(current_closest_distance)
            if len(distance_history) > DISTANCE_BUFFER_SIZE:
                distance_history.pop(0) 

            # Calculate smoothed distance using the median of the history
            closest_distance = np.median(distance_history)
        else:
            distance_history = []
            closest_distance = float('inf') 

        # --- D. Console Output Decision ---
        approaching_person_count = len(person_detections) 

        if approaching_person_count > 0 and closest_distance < float('inf'):
            
            if current_time - last_alert_time > ALERT_COOLDOWN:
                is_initial = (LAST_SPOKEN_STATUS != "approaching")
                distance_changed = abs(closest_distance - last_spoken_distance) > DISTANCE_CHANGE_THRESHOLD
                count_changed = approaching_person_count != last_spoken_count
                
                if is_initial or distance_changed or count_changed:
                    person_text = "1 person" if approaching_person_count == 1 else f"{approaching_person_count} people"
                    alert_text = f"Warning, {person_text} approaching. Closest: {closest_distance:.1f} meters."
                    
                    speak(alert_text)
                    last_alert_time = current_time
                    LAST_SPOKEN_STATUS = "approaching"
                    last_spoken_distance = closest_distance
                    last_spoken_count = approaching_person_count
                        
        else:
            if LAST_SPOKEN_STATUS == "approaching":
                LAST_SPOKEN_STATUS = None
                last_spoken_distance = float('inf')
                last_spoken_count = 0
            
            if current_time - last_alert_time > 15 and LAST_SPOKEN_STATUS != "clear":
                 speak("Path is clear.")
                 LAST_SPOKEN_STATUS = "clear"
                 last_alert_time = current_time 

        cv2.imshow("Blind Assistant View", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- CLEANUP ---
    cap.release()
    cv2.destroyAllWindows()
    speak("System shutting down.")


if __name__ == "__main__":
    if not os.path.exists("basic_detector_model.pth"):
        print("\n*** ERROR: Model file 'basic_detector_model.pth' not found. ***")
        print("Please ensure your trained .pth file is in the same directory as this script.")
    else:
        main_inference()