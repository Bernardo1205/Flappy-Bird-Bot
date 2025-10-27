import cv2
import numpy as np
from ultralytics import YOLO
from mss import mss
import time
import pyautogui

# Load your trained YOLOv8 model
model = YOLO('runs/detect/train4/weights/best.pt')  # Replace with your model path


print("You have 5 seconds to focus the game window...")
time.sleep(5)

# Get the active window position and size
window = pyautogui.getActiveWindow()

# Set up screen capture (adjust monitor coordinates to match your game window)
monitor = {"top": window.top, "left": window.left, "width": window.width, "height": window.height}  # Adjust to your game window size

# Initialize screen capture
sct = mss()

# For FPS calculation
prev_time = 0

window_name = "Flappy Bird Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("Starting detection. Press 'q' to exit.")

while True:
    # Capture screen
    screenshot = sct.grab(monitor)
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Run YOLOv8 inference
    results = model.predict(img) # Adjust confidence threshold as needed

    # Visualize results
    annotated_frame = results[0].plot()

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Display FPS
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Update the existing window instead of creating a new one
    cv2.imshow(window_name, annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()