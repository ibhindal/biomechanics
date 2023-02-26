from absl import flags
import sys
FLAGS=flags.FLAGS
FLAGS(sys.argv)

import time 
import numpy as np
import os
import cv2

import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import   YoloV3
from yolov3_tf2.datasets import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import detection
from deep_sort.tracker import tracker
from tools import generate_detections as gdet


class_names=[s.strip() for c in open('./data/labels/coco.names').readlines()]
yolo=yoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

max_cosine_distance = 0.5
nnbudget = None
