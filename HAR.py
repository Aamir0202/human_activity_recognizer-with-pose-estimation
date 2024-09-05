import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from ultralytics import YOLO

# Load the pre-trained I3D model from TensorFlow Hub
model_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
i3d_model = hub.load(model_url)
i3d_model = i3d_model.signatures['default']

# Load Kinetics-400 class names
class_names_url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
class_names_path = tf.keras.utils.get_file("label_map.txt", class_names_url)
with open(class_names_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Function to preprocess video frames for I3D
def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0  # Normalize to [0, 1]
    return frame

# Function to predict top 5 actions and return the most predicted action
def predict_action(sample_video):
    # Add a batch axis to the sample video
    model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

    logits = i3d_model(model_input)['default'][0]
    probabilities = tf.nn.softmax(logits)

    top_actions = []
    for i in np.argsort(probabilities)[::-1][:5]:
        top_actions.append((class_names[i], probabilities[i].numpy()))
    
    # Print top 5 actions in the terminal
    print("\nTop 5 Actions:")
    for idx, (action, prob) in enumerate(top_actions):
        print(f"{idx + 1}. {action}: {prob * 100:.2f}%")
    
    # Return the most predicted action (highest probability)
    return top_actions[0]  # Most probable action

# Load the YOLOv8 pose model
yolo_model = YOLO('yolov8n-pose.pt')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Buffer to store video frames
frame_buffer = []
buffer_size = 16  # Number of frames to consider for each prediction

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for I3D and add to buffer
    preprocessed_frame = preprocess_frame(frame)
    frame_buffer.append(preprocessed_frame)

    # Run YOLOv8 pose detection
    yolo_result = yolo_model(frame, show=False)
    annotated_frame = yolo_result[0].plot()

    # If buffer is full, run action recognition prediction
    if len(frame_buffer) == buffer_size:
        most_predicted_action, prob = predict_action(frame_buffer)
        frame_buffer = []  # Clear the buffer

        # Overlay the most predicted action above each detected person (bounding box)
        for detection in yolo_result[0].boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
            label = f"{most_predicted_action}: {prob * 100:.2f}%"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the combined frame with YOLOv8 pose detection and action recognition results
    cv2.imshow('Real-Time Action Recognition and Pose Detection', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
