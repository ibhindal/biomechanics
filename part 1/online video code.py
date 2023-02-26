import os
import time
import tensorflow as tf
import cv2
import numpy as np
from utils import label_map_util
from PIL import Image
from google.colab.patches import cv2_imshow
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from IPython.display import HTML
from base64 import b64encode

# Path to saved model
PATH_TO_SAVED_MODEL = "/content/gdrive/MyDrive/customTF2/data/inference_graph/saved_model"

# Load label map and obtain class names and ids
category_index = label_map_util.create_category_index_from_labelmap("/mydrive/customTF2/data/label_map.pbtxt", use_display_name=True)

def visualise_on_image(image, bboxes, labels, scores, thresh):
    """Draws bounding boxes and labels on image"""
    (h, w, d) = image.shape
    for bbox, label, score in zip(bboxes, labels, scores):
        if score > thresh:
            xmin, ymin = int(bbox[1]*w), int(bbox[0]*h)
            xmax, ymax = int(bbox[3]*w), int(bbox[2]*h)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
            cv2.putText(image, f"{label}: {int(score*100)} %", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return image

def detect_objects_in_image(image, detect_fn, score_thresh, max_detections):
    """Returns detections for image using detect_fn"""
    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(image)[tf.newaxis, ...]

    # Pass image through detector
    detections = detect_fn(input_tensor)

    # Get scores, bboxes, and labels from detections
    scores = detections['detection_scores'][0, :max_detections].numpy()
    bboxes = detections['detection_boxes'][0, :max_detections].numpy()
    labels = detections['detection_classes'][0, :max_detections].numpy().astype(np.int64)

    # Filter detections based on score threshold
    filtered_scores = scores[scores > score_thresh]
    filtered_bboxes = bboxes[:len(filtered_scores)]
    filtered_labels = labels[:len(filtered_scores)]

    return filtered_bboxes, filtered_labels, filtered_scores

if __name__ == '__main__':
    # Load the model
    print("Loading saved model ...")
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    print("Model Loaded!")

    # Video Capture (video_file)
    video_capture = cv2.VideoCapture("/mydrive/input.mp4")
    start_time = time.time()

    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    #fps = int(video_capture.get(5))
    size = (frame_width, frame_height)

    # Initialize video writer
    result = cv2.VideoWriter('/mydrive/result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15, size)

while True:
    # Read frame from video
    ret, frame = video_capture.read()
    if not ret:
        print('Unable to read video / Video ended')
        break

    # Flip frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB
    image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get detections for image
    bboxes, labels, scores = detect_objects_in_image(image_np, detect_fn, score_thresh=0.4, max_detections=1)

    # Visualize detections on image
    image_np = visualise_on_image(image_np, bboxes, labels, scores, thresh=0.4)
# Write frame to output video
    result.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))