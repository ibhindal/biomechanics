import os
import cv2

# Set the input and output directories
input_dir = 'C:/Users/Ibrahim/Desktop/images'
output_dir = 'C:/Users/Ibrahim/Desktop/bw'

# Iterate over all the files in the input directory
for file in os.listdir(input_dir):
  # Load the image
  image = cv2.imread(os.path.join(input_dir, file))
  
  # Convert the image to grayscale
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  # Save the grayscale image to the output directory
  cv2.imwrite(os.path.join(output_dir, file), gray_image)
