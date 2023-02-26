#trial with tensorflow


# Import required libraries
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the TensorFlow model
model = tf.keras.models.load_model('ball_detection_model.h5')

# Capture the video
cap = cv2.VideoCapture('video.mp4')

# Get the first frame of the video
ret, prev_frame = cap.read()

# Convert the frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Initialize lists to store the ball position and speed
positions = []
speeds = []

# Loop over each frame of the video
while True:
    # Read the current frame
    ret, frame = cap.read()
    
    # Check if we have reached the end of the video
    if not ret:
        break
    
    # Use the TensorFlow model to detect the ball in the frame
    ball = model.predict(frame)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use the Lucas-Kanade algorithm to compute the optical flow between the previous and current frames
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Use the optical flow and the detected ball to predict the position and speed of the ball in the next frame
    # (This step will require some additional code, which is beyond the scope of this example)
    position = predict_position(ball, flow)
    speed = predict_speed(position, positions)
    
    # Use the predicted position to guide a bounding box around the ball in the next frame
    # (This step will also require additional code)
    
    # Show the frame with the bounding box
    cv2.imshow('Frame', frame)
    
    # Check if the user has pressed the 'q' key to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Update the previous frame and grayscale image, and append the current position and speed to the lists
    prev_frame = frame
    prev_gray = gray
    positions.append(position)
    speeds.append(speed)

# Plot the ball speed on a graph
plt.plot(speeds)
plt.show()

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
