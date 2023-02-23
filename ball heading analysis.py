# ball heading analysis
#!python detect_or_track.py --weights yolov7_custom.pt --source CL_1_S0003.mp4 --save-txt --name CL_1_S0003
import numpy as np
import os
import pandas as pd
import scipy
import math


# Specify the folder containing the text files
folder_path = "C:/Users/Ibrahim/biomechanics/yolov7/runs/detect/" + filesname + "/labels"

# Create an empty list to store the data from each file
all_data = []

# Iterate through all the files in the folder
for filename in sorted(os.listdir(folder_path), key=lambda x: int(x.split("_")[-1].rstrip(".txt"))):
    if filename.endswith(".txt"):
        # Open the file and read its contents
        with open(os.path.join(folder_path, filename), 'r') as file:
            file_data = file.readline()

        # Split the line into a list of numbers
        line_data = file_data.strip().split(" ")

        # Convert the numbers to integers
        line_data = [float(x) for x in line_data]

        # Add the line data to the list of all data
        all_data.append(line_data)

# Convert the list of data to a pandas DataFrame
df = pd.DataFrame(all_data)
display(df)

df.to_csv('filename.csv', index=False)

# Write the DataFrame to a CSV file in the same folder as the text files
df.to_csv(os.path.join(folder_path, "concatenated_data.csv"), index=False)


# Set the directory to the current directory #os.chdir("C:\\Users\\Ibrahim\\desktop")
file= videopath #for name currently


x_wall = xy #coordinates of the wall, uses the global variables from the other script
y_wall = yy

num_cont_frames=0
num_count_inbound = 0
#video = cv.VideoCapture(path)
ball_size = 0.22  #diameter of a regulation ball in meters
fps_cam = 10000  # Change this to the required fps of the video

# Initialize variable to track state of ball (in contact with wall or not)
in_contact = False

# Initialize variable to track whether in_contact has ever been True
in_contact_ever = False


# Initialize lists for inbound and outbound velocities
inbound_velocities = []
outbound_velocities = []

# Calculate time interval between frames in seconds
time_interval = 1 / fps_cam

scale  = []
x_def = []
inbound_x = []
inbound_y = []
outbound_x = []
outbound_y = []
threshold = 0.05 # adjust this value based on your data


for i, (index, row) in enumerate(df.iterrows()): #change this to for each line of the pandas df
#x is x center y is y center
    x, y, w, h = row[1], row[2], row[3], row[4]
    x2=x+w/2
    y2=y+h/2
    x1=x-w/2
    y1=y-h/2
    
    
    scale.append(ball_size/w)   #meters per pixel.diameter in pixels or coordinate value / real diameter in m to give pixel per m for a scale factor
    
    if x1 < x_wall: #sometimes the bbox is the wrong way around
        # Set in_contact to True
        in_contact = True
        # Set in_contact_ever to True
        in_contact_ever = True
        # Increment counter
        num_cont_frames = num_cont_frames + 1
        x_defe = x2-x_wall
        x_def.append(x_defe)
        
        
        
    else:
        in_contact = False

    if in_contact == False and in_contact_ever==False:
        num_count_inbound  = num_count_inbound  + 1

    if in_contact == False and in_contact_ever==True:
        outbound_x.append(x2) #list of x positions of right edge
        outbound_y.append(y2)
  
    if in_contact == True:
        if i<len(df)-1:
            next_row = df.iloc[i+1]
            next_x1 = next_row[1] - next_row[3]/2
            if abs(x1 - next_x1) > threshold:
                df.drop(index+1, inplace=True)
                continue
    
# Modifying the number of rows to take from the dataframe if num_count_inbound is less than 10
if num_count_inbound < 10:
    num_count_inbound += 1

# Taking the specified number of rows from the dataframe
result = df.head(num_count_inbound)

# Calculating x2 and y2 for each row in the result dataframe
for index, row in result.iterrows():
    x2 = row[1] + 0.5 * row[3]
    y2 = row[2] + 0.5 * row[4]
    
    inbound_x.append(x2)
    inbound_y.append(y2)
 
scale_ave=scipy.stats.trim_mean(scale, 0.3) #trim_mean 30% either way to remove some extraneous results

display(df)
df.to_csv('filename.csv', index=False)


contact_time = num_cont_frames/fps_cam

if x_def:
    realxdef = min(x_def)*scale_ave
else:
    realxdef = 0



inbound_x_diff = []
inbound_y_diff = []

inbound_x_velocities = []
inbound_y_velocities = []
inbound_len = len(inbound_x) - 1


# Calculate differences between consecutive x and y coordinates
for i in range(inbound_len):
    inbound_x_diff.append(inbound_x[i] - inbound_x[i + 1])
    inbound_y_diff.append(inbound_y[i] - inbound_y[i + 1])


filtered_inbound_x_diff = []
filtered_inbound_y_diff = []

for i in range(inbound_len):
    if inbound_x_diff[i] <= 1:
        filtered_inbound_x_diff.append(inbound_x_diff[i])
inbound_x_diff = filtered_inbound_x_diff


for i in range(inbound_len):
    if inbound_y_diff[i] <= 1:
        filtered_inbound_y_diff.append(inbound_y_diff[i])
inbound_y_diff = filtered_inbound_y_diff

inbound_y_diff = [abs(value) for value in inbound_y_diff]
inbound_x_diff = [abs(value) for value in inbound_x_diff]

# Calculate inbound velocities in meters per second
inbound_velocities = []
for i in range(inbound_len):
    inbound_x_velocity = inbound_x_diff[i] * scale_ave * fps_cam
    inbound_x_velocities.append(inbound_x_velocity)
    print(inbound_x_velocity)
    inbound_y_velocity = inbound_y_diff[i] * scale_ave * fps_cam
    inbound_y_velocities.append(inbound_y_velocity)

# Calculate outbound velocities
outbound_x_diff = []
outbound_y_diff = []

outbound_len = len(outbound_x) - 1

# Calculate differences between consecutive x and y coordinates
for i in range(outbound_len):
    outbound_x_diff.append(outbound_x[i] - outbound_x[i + 1])
    outbound_y_diff.append(outbound_y[i] - outbound_y[i + 1])
    
outbound_y_diff = [abs(value) for value in outbound_y_diff]
outbound_x_diff = [abs(value) for value in outbound_x_diff]


    
filtered_outbound_x_diff = []
filtered_outbound_y_diff = []

for i in range(outbound_len):
    if outbound_x_diff[i] <= 1:
        filtered_outbound_x_diff.append(outbound_x_diff[i])
outbound_x_diff = filtered_outbound_x_diff

for i in range(outbound_len):
    if outbound_y_diff[i] <= 1:
        filtered_outbound_y_diff.append(outbound_y_diff[i])
outbound_y_diff = filtered_outbound_y_diff


# Calculate outbound velocities in meters per second
outbound_velocities = []
outbound_x_velocities = []
outbound_y_velocities = []

for i in range(outbound_len):
    outbound_x_velocity = outbound_x_diff[i] * scale_ave * fps_cam
    outbound_x_velocities.append(outbound_x_velocity)
    outbound_y_velocity = outbound_y_diff[i] * scale_ave * fps_cam
    outbound_y_velocities.append(outbound_y_velocity)


  
    
filtered_outbound_x_velocities = []
for value in outbound_x_velocities:
    if value < 100:
        filtered_outbound_x_velocities.append(value)
outbound_x_velocities = filtered_outbound_x_velocities

filtered_outbound_y_velocities = []
for value in outbound_y_velocities:
    if value < 100:
        filtered_outbound_y_velocities.append(value)
outbound_y_velocities = filtered_outbound_y_velocities

filtered_inbound_x_velocities = []
for value in inbound_x_velocities:
    if value < 100:
        filtered_inbound_x_velocities.append(value)
inbound_x_velocities = filtered_inbound_x_velocities

filtered_inbound_y_velocities = []
for value in inbound_y_velocities:
    if value < 100:
        filtered_inbound_y_velocities.append(value)
inbound_y_velocities = filtered_inbound_y_velocities


corrected_average_inbound_x_velocities = scipy.stats.trim_mean(inbound_x_velocities, 0.2)
corrected_average_inbound_y_velocities = scipy.stats.trim_mean(inbound_y_velocities, 0.2)
corrected_average_inbound_velocities = math.hypot(corrected_average_inbound_y_velocities, corrected_average_inbound_x_velocities)
corrected_average_outbound_x_velocities = scipy.stats.trim_mean(outbound_x_velocities, 0.2)
corrected_average_outbound_y_velocities = scipy.stats.trim_mean(outbound_y_velocities, 0.2)
corrected_average_outbound_velocities = math.hypot(corrected_average_outbound_y_velocities, corrected_average_outbound_x_velocities)


# Create a dictionary with the data for the table
speeddata={'inbound x velocity ': corrected_average_inbound_x_velocities, 'inbound y velocity': corrected_average_inbound_y_velocities, 'inbound velocity': corrected_average_inbound_velocities, 'outbound x velocity': corrected_average_outbound_x_velocities, 'outbound y velocity': corrected_average_outbound_y_velocities, 'outbound velocity': corrected_average_outbound_velocities, 'contact time': contact_time, 'deformation' :realxdef}

df2 = pd.DataFrame(speeddata, index=range(len(speeddata)))

# Export the DataFrame to a CSV file
filename = file + 'results.csv'
df2.to_csv(filename, index=False)