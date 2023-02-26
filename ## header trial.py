## biomechanics yolo
#!python detect_or_track.py --weights yolov7_custom.pt --source CL_1_S0003.mp4 --save-txt --name CL_1_S0003
import numpy as np
import os
import pandas as pd
import scipy
import math
import scipy.stats

def perform_analysis(df, fps_cam, scale_ave):
    # Calculate contact time
    num_cont_frames = len(df)
    contact_time = num_cont_frames / fps_cam
    
    # Calculate real x deflection
    x_def = df['x_def'].tolist()
    if x_def:
        realxdef = min(x_def) * scale_ave
    else:
        realxdef = 0
    
    # Calculate inbound velocities
    inbound_x = df['inbound_x'].tolist()
    inbound_y = df['inbound_y'].tolist()
    inbound_x_diff = [abs(inbound_x[i] - inbound_x[i + 1]) for i in range(len(inbound_x) - 1)]
    inbound_y_diff = [abs(inbound_y[i] - inbound_y[i + 1]) for i in range(len(inbound_y) - 1)]
    filtered_inbound_x_diff = [val for val in inbound_x_diff if val <= 1]
    filtered_inbound_y_diff = [val for val in inbound_y_diff if val <= 1]
    inbound_x_diff = filtered_inbound_x_diff
    inbound_y_diff = filtered_inbound_y_diff
    inbound_y_diff = [abs(value) for value in inbound_y_diff]
    inbound_x_diff = [abs(value) for value in inbound_x_diff]
    inbound_velocities = [(inbound_x_diff[i] * scale_ave * fps_cam, inbound_y_diff[i] * scale_ave * fps_cam) for i in range(len(inbound_x_diff))]
    inbound_x_velocities = [velocity[0] for velocity in inbound_velocities]
    inbound_y_velocities = [velocity[1] for velocity in inbound_velocities]

    # Calculate outbound velocities
    outbound_x = df['outbound_x'].tolist()
    outbound_y = df['outbound_y'].tolist()
    outbound_x_diff = [abs(outbound_x[i] - outbound_x[i + 1]) for i in range(len(outbound_x) - 1)]
    outbound_y_diff = [abs(outbound_y[i] - outbound_y[i + 1]) for i in range(len(outbound_y) - 1)]
    filtered_outbound_x_diff = [val for val in outbound_x_diff if val <= 1]
    filtered_outbound_y_diff = [val for val in outbound_y_diff if val <= 1]
    outbound_x_diff = filtered_outbound_x_diff
    outbound_y_diff = filtered_outbound_y_diff
    outbound_y_diff = [abs(value) for value in outbound_y_diff]
    outbound_x_diff = [abs(value) for value in outbound_x_diff]
    outbound_velocities = [(outbound_x_diff[i] * scale_ave * fps_cam, outbound_y_diff[i] * scale_ave * fps_cam) for i in range(len(outbound_x_diff))]
    outbound_x_velocities = [velocity[0] for velocity in outbound_velocities]
    outbound_y_velocities = [velocity[1] for velocity in outbound_velocities]

    # Filter velocities
    filtered_outbound_x_velocities = [value for value in outbound_x_velocities if value < 100]
    filtered_outbound_y_velocities = [value for value in outbound_y_velocities if value < 100]
    filtered_inbound_x_velocities = [value for value in inbound_x_velocities if value < 100]
    filtered_inbound_y_velocities = [value for value in inbound_y_velocities if value < 100]
    #Trim the inbound velocities using scipy.stats.trim_mean
    #Use the average of the trimmed values as the corrected average velocity
    corrected_average_inbound_x_velocities = scipy.stats.trim_mean(inbound_x_velocities, 0.2)
    corrected_average_inbound_y_velocities = scipy.stats.trim_mean(inbound_y_velocities, 0.2)

    #Calculate the corrected average inbound velocity as the Euclidean distance of x and y components
    corrected_average_inbound_velocities = math.hypot(corrected_average_inbound_y_velocities, corrected_average_inbound_x_velocities)

    #Trim the outbound velocities using scipy.stats.trim_mean
    #Use the average of the trimmed values as the corrected average velocity
    corrected_average_outbound_x_velocities = scipy.stats.trim_mean(outbound_x_velocities, 0.2)
    corrected_average_outbound_y_velocities = scipy.stats.trim_mean(outbound_y_velocities, 0.2)

    #Calculate the corrected average outbound velocity as the Euclidean distance of x and y components
    corrected_average_outbound_velocities = math.hypot(corrected_average_outbound_y_velocities, corrected_average_outbound_x_velocities)

    #Calculate the average inbound and outbound velocity
    average_inbound_velocity = sum(inbound_velocities) / inbound_len
    average_outbound_velocity = sum(outbound_velocities) / outbound_len

    #Calculate the standard deviation of the inbound and outbound velocities
    std_inbound_velocity = statistics.stdev(inbound_velocities)
    std_outbound_velocity = statistics.stdev(outbound_velocities)

    #Print the results
    print(f"Corrected Average Inbound Velocity: {corrected_average_inbound_velocities:.2f} m/s")
    print(f"Corrected Average Outbound Velocity: {corrected_average_outbound_velocities:.2f} m/s")
    print(f"Average Inbound Velocity: {average_inbound_velocity:.2f} m/s")
    print(f"Average Outbound Velocity: {average_outbound_velocity:.2f} m/s")
    print(f"Standard Deviation Inbound Velocity: {std_inbound_velocity:.2f} m/s")
    print(f"Standard Deviation Outbound Velocity: {std_outbound_velocity:.2f} m/s")


    return (df, {
        'contact_time': contact_time,
        'realxdef': realxdef,
        'inbound_velocities': inbound_velocities,
        'outbound_velocities': outbound_velocities,
        'corrected_average_inbound_velocities': corrected_average_inbound_velocities,
        'corrected_average_outbound_velocities': corrected_average_outbound_velocities
    })




x_head = xy #coordinates of the head, uses the global variables from the other script
y_head = yy

num_cont_frames=0
num_count_inbound = 0
#video = cv.VideoCapture(path)
ball_size = 0.22  #diameter of a regulation ball in meters
fps_cam = 10000  # Change this to the required fps of the video

# Initialize variable to track state of ball (in contact with head or not)
in_contact = False

# Initialize variable to track whether in_contact has ever been True
in_contact_ever = False

# a list containg the names of the dataframes
dfs = []

#initialise variable to get data from 4 files
file_names = ['ball1', 'ball2', 'head1', 'head2']

for i, file_name in enumerate(file_names):
    # Specify the folder containing the text files
    folder_path = "C:/Users/Ibrahim/biomechanics/yolov7/runs/detect/" + file_name + "/labels"

    # Create an empty list to store the data from each file
    all_data = []

    # Iterate through all the files in the folder
    for filename in sorted(os.listdir(folder_path), key=lambda x: int(x.split("_")[-1].rstrip(".txt"))):
        if filename.endswith(".txt"):
            # Open the file and read its contents
            with open(os.path.join(folder_path, filename), 'r') as file:
                file_data = file.read()

            # Split the data into a list of lines
            lines = file_data.strip().split('\n')

            # Split each line into a list of numbers and convert to floats
            for line in lines:
                line_data = line.strip().split(' ')
                line_data = [file_name] + [float(x) for x in line_data]
                all_data.append(line_data)

    # Convert the list of data to a pandas DataFrame
    df = pd.DataFrame(all_data, columns=['file_name', 'class', 'x', 'y', 'w', 'h'])

    # Add the DataFrame to the list of DataFrames
    dfs.append(df)

    # Calculate the bounding box coordinates and add them as new columns
    df['x1'] = df['x'] - df['w'] / 2
    df['y1'] = df['y'] - df['h'] / 2
    df['x2'] = df['x'] + df['w'] / 2
    df['y2'] = df['y'] + df['h'] / 2

    # Save the DataFrame to a CSV file with a unique name
    csv_file_name = file_name + '_concatenated.csv'
    df.to_csv(os.path.join(folder_path, csv_file_name), index=False)


    # Set the directory to the current directory #os.chdir("C:\\Users\\Ibrahim\\desktop")
    file= filesname #for name currently


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


    for i, df in enumerate(dfs): # Iterate over each DataFrame in the list
        for j, (index, row) in enumerate(df.iterrows()): # Iterate over each row in the DataFrame
            #x is x center y is y center
            x, y, w, h = row[1], row[2], row[3], row[4]
            x2=x+w/2
            y2=y+h/2
            x1=x-w/2
            y1=y-h/2

            # Add the file name before x, y, w, h
            file_name = f'file{i+1}'
            row = [file_name] + row.tolist()

            scale.append(ball_size/w)   #meters per pixel.diameter in pixels or coordinate value / real diameter in m to give pixel per m for a scale factor

            if x1 < x_head: #sometimes the bbox is the wrong way around
                # Set in_contact to True
                in_contact = True
                # Set in_contact_ever to True
                in_contact_ever = True
                # Increment counter
                num_cont_frames = num_cont_frames + 1
                x_defe = x2-x_head
                x_def.append(x_defe)

            else:
                in_contact = False

            if in_contact == False and in_contact_ever==False:
                num_count_inbound  = num_count_inbound  + 1

            if in_contact == False and in_contact_ever==True:
                outbound_x.append(x2) #list of x positions of right edge
                outbound_y.append(y2)

            if in_contact == True:
                if j<len(df)-1:
                    next_row = df.iloc[j+1]
                    next_x1 = next_row[1] - next_row[3]/2
                    if abs(x1 - next_x1) > threshold:
                        df.drop(index+1, inplace=True)
                        continue

        # Modifying the number of rows to take from the dataframe if num_count_inbound is less than 10
        if num_count_inbound < 10:
            num_count_inbound += 1

        # Taking the specified number of rows from the dataframe
        result = df.head(num_count_inbound)
        dfs[i] = result # replace the original dataframe with the truncated version


        # Calculating x2 and y2 for each row in the result dataframe
        for index, row in result.iterrows():
            x2 = row[2] + 0.5 * row[4]
            y2 = row[3] + 0.5 * row[5]

            inbound_x.append(x2)
            inbound_y.append(y2)


scale_ave=scipy.stats.trim_mean(scale, 0.3) #trim_mean 30% either way to remove some extraneous results

#display(df)
#df.to_csv('filename.csv', index=False)



all_results = []
for i, df in enumerate([df1, df2, df3, df4]):
    results = perform_analysis(df, fps_cam, scale_ave)
    all_results.append(results)
    print(f"Results for dataframe {i+1}: {results}")




# Create a dictionary with the data for the table
speeddata={'inbound x velocity ': corrected_average_inbound_velocities, 'inbound y velocity': corrected_average_inbound_y_velocities, 'inbound velocity': corrected_average_inbound_velocities, 'outbound x velocity': corrected_average_outbound_x_velocities, 'outbound y velocity': corrected_average_outbound_y_velocities, 'outbound velocity': corrected_average_outbound_velocities, 'contact time': contact_time, 'deformation' :realxdef}

df2 = pd.DataFrame(speeddata, index=range(len(speeddata)))

# Export the DataFrame to a CSV file
filename = file + 'results.csv'
df2.to_csv(filename, index=False)