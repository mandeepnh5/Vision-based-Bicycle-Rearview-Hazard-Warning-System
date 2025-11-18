import cv2
from ultralytics import YOLO

# --- Configuration ---
# 1. Load the pre-trained deer detection model from Hugging Face
# This model is specifically fine-tuned for deer detection.
model = YOLO("deer.pt")

# 2. Set the path to your video file
VIDEO_PATH = "deer/1.mp4"  # <-- IMPORTANT: Change this

# 3. (Optional) Print the model's class names to find the 'deer' ID
# This helps you verify the class ID. In this model, 'deer' is typically class 0.
print("Model classes:", model.names)
DEER_CLASS_ID = 0  # Assuming 'deer' is class 0, adjust if model.names shows differently
# --- End Configuration ---


# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 detection on the frame, filtering for the 'deer' class only
        # stream=True is more memory-efficient for video
        results = model(frame, stream=True, classes=[DEER_CLASS_ID])

        # The 'results' object is a generator
        for r in results:
            # Get the annotated frame (with boxes and labels)
            annotated_frame = r.plot()

            # Display the annotated frame in a window
            cv2.imshow("Deer Detection", annotated_frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the video has ended
        break

# Release the video capture object and close the display windows
cap.release()
cv2.destroyAllWindows()

print("Video processing finished.")