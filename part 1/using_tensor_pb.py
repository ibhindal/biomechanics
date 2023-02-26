# Import necessary modules
import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# Load the .pb file
model = tf.saved_model.load("C:\\Users\\Ibrahim\\biomechanics")


# Load the label map
label_map = label_map_util.load_labelmap('saved_model.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Create a video capture object
cap = cv2.VideoCapture('P2_1_30_1.mp4')

# Loop until the end of the video
while cap.isOpened():
  # Read the next frame from the video
  ret, frame = cap.read()

  # If the frame was not successfully read, break the loop
  if not ret:
    break

  # Perform object detection
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      # Define input and output tensors (i.e. data) for the object detection classifier
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
      detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      # Expand frame dimensions to have shape: [1, None, None, 3]
      # i.e. a single-column array, where each item in the column has the pixel RGB value
      frame_expanded = np.expand_dims(frame, axis=0)

      # Perform the actual detection by running the model with the frame as input
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: frame_expanded})

      # Draw the results of the detection (aka 'visulaize the results')
      vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)

    # Show the output frame
    cv2.imshow('Object detector', frame)

    # Wait for a key press
    key = cv2.waitKey(1)
    if key == 27:  # escape key
      break