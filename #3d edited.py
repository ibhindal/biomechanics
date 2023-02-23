#3d edited

import math

# Camera positions in 3D space
camera1_position = [-2, 0, 4]
camera2_position = [2, 0, 4]

fps = 10000
time_per_frame = 1/fps

# Distance from cameras to target
d_camera = 4

# Distance between cameras
d_between_cameras = 0.5

# Angle between cameras in degrees
angle_between_cameras = math.degrees(math.atan(d_between_cameras/d_camera))

#initial diameter of ball
initial_diameter = 0.22 # meters




# Load data from input files
with open('ball_camera1.txt', 'r') as f:
    ball_camera1_data = f.readlines()
with open('ball_camera2.txt', 'r') as f:
    ball_camera2_data = f.readlines()
with open('head_camera1.txt', 'r') as f:
    head_camera1_data = f.readlines()
with open('head_camera2.txt', 'r') as f:
    head_camera2_data = f.readlines()

# Convert data to lists of floats
x_ball_camera1_norm = [float(line.split(',')[0]) for line in ball_camera1_data]
y_ball_camera1_norm = [float(line.split(',')[1]) for line in ball_camera1_data]
w_ball_camera1_norm = [float(line.split(',')[2]) for line in ball_camera1_data]
h_ball_camera1_norm = [float(line.split(',')[3]) for line in ball_camera1_data]

x_ball_camera2_norm = [float(line.split(',')[0]) for line in ball_camera2_data]
y_ball_camera2_norm = [float(line.split(',')[1]) for line in ball_camera2_data]
w_ball_camera2_norm = [float(line.split(',')[2]) for line in ball_camera2_data]
h_ball_camera2_norm = [float(line.split(',')[3]) for line in ball_camera2_data]

x_head_camera1_norm = [float(line.split(',')[0]) for line in head_camera1_data]
y_head_camera1_norm = [float(line.split(',')[1]) for line in head_camera1_data]
w_head_camera1_norm = [float(line.split(',')[2]) for line in head_camera1_data]
h_head_camera1_norm = [float(line.split(',')[3]) for line in head_camera1_data]

x_head_camera2_norm = [float(line.split(',')[0]) for line in head_camera2_data]
y_head_camera2_norm = [float(line.split(',')[1]) for line in head_camera2_data]
w_head_camera2_norm = [float(line.split(',')[2]) for line in head_camera2_data]
h_head_camera2_norm = [float(line.split(',')[3]) for line in head_camera2_data]

# Convert xywh to x, y, z for ball in camera 1
x_ball_camera1_norm = [(x + w/2 - 0.5) for x, w in zip(x_ball_camera1_norm, w_ball_camera1_norm)]
y_ball_camera1_norm = [(y + h/2 - 0.5) for y, h in zip(y_ball_camera1_norm, h_ball_camera1_norm)]
z_ball_camera1 = [d_camera for _ in range(len(x_ball_camera1_norm))]

# Convert xywh to x, y, z for ball in camera 2
x_ball_camera2_norm = [(x + w/2 - 0.5) for x, w in zip(x_ball_camera2_norm, w_ball_camera2_norm)]
y_ball_camera2_norm = [(y + h/2 - 0.5) for y, h in zip(y_ball_camera2_norm, h_ball_camera2_norm)]
z_ball_camera2 = [d_camera for _ in range(len(x_ball_camera2_norm))]


# Convert distances to meters
x_ball_camera1 = [d * pixels_to_meters for d in x_ball_camera1_norm]
y_ball_camera1 = [d * pixels_to_meters for d in y_ball_camera1_norm]
x_ball_camera2 = [d * pixels_to_meters for d in x_ball_camera2_norm]
y_ball_camera2 = [d * pixels_to_meters for d in y_ball_camera2_norm]

# Calculate the position of the ball in 3D space
# First, calculate the distance from each camera to the ball at each frame
d_camera1 = [math.sqrt((x_ball_camera1_norm[i] - camera1_position[0])**2 + (y_ball_camera1_norm[i] - camera1_position[1])**2) for i in range(len(x_ball_camera1_norm))]
d_camera2 = [math.sqrt((x_ball_camera2_norm[i] - camera2_position[0])**2 + (y_ball_camera2_norm[i] - camera2_position[1])**2) for i in range(len(x_ball_camera2_norm))]

# Next, calculate the angle between the two cameras
angle_between_cameras = math.degrees(math.atan2(camera2_position[0] - camera1_position[0], camera2_position[2] - camera1_position[2]))

# Calculate the position of the ball in 3D space
x_ball = []
y_ball = []
z_ball = []
for i in range(len(x_ball_camera1_norm)):
    # Convert distances to meters
    d_camera1_m = d_camera1[i] * pixels_to_meters
    d_camera2_m = d_camera2[i] * pixels_to_meters
    
    # Calculate the x, y, and z coordinates of the ball in 3D space
    x_ball_camera1_norm = x_ball_camera1_norm[i] - camera1_position[0]
    y_ball_camera1_norm = y_ball_camera1_norm[i] - camera1_position[1]
    z_ball_camera1 = d_camera1_m
    
    x_ball_camera2_norm = x_ball_camera2_norm[i] - camera2_position[0]
    y_ball_camera2_norm = y_ball_camera2_norm[i] - camera2_position[1]
    z_ball_camera2 = d_camera2_m
    
    x_ball.append((camera1_position[2] - camera2_position[2]) / math.tan(math.radians(angle_between_cameras)) * (x_ball_camera1_norm - x_ball_camera2_norm))
    y_ball.append((camera1_position[2] - camera2_position[2]) / math.tan(math.radians(angle_between_cameras)) * (y_ball_camera1_norm - y_ball_camera2_norm))
    z_ball.append((camera1_position[2] - camera2_position[2]) / math.tan(math.radians(angle_between_cameras)) * (z_ball_camera1 - z_ball_camera2))

# Calculate the velocity of the ball at each frame
x_ball_velocity = []
y_ball_velocity = []
z_ball_velocity = []

for i in range(1, len(x_ball)):
    x_ball_velocity.append((x_ball[i] - x_ball[i-1]) / time_per_frame)
    y_ball_velocity.append((y_ball[i] - y_ball[i-1]) / time_per_frame)
    z_ball_velocity.append((z_ball[i] - z_ball[i-1]) / time_per_frame)

# Calculate the position of the head in 3D space
x_head = []
y_head = []
z_head = []

# Convert head coordinates from camera 1 to 3D space
x_head_camera1_norm_m = [(x + w/2) / 100 for (x, _, w, _) in x_head_camera1_norm[i]]
y_head_camera1_norm_m = [(y + h/2) / 100 for (_, y, _, h) in x_head_camera1_norm[i]]
x_head_camera1 = [(x - camera1_position[0]) * d_camera1_m / camera1_focal_length for x in x_head_camera1_norm_m]
y_head_camera1 = [(y - camera1_position[1]) * d_camera1_m / camera1_focal_length for y in y_head_camera1_norm_m]
z_head_camera1 = [d_camera1_m for _ in range(len(x_head_camera1_norm_m))]

# Convert head coordinates from camera 2 to 3D space
x_head_camera2_norm_m = [(x + w/2) / 100 for (x, _, w, _) in x_head_camera2_norm[i]]
y_head_camera2_norm_m = [(y + h/2) / 100 for (_, y, _, h) in x_head_camera2_norm[i]]
x_head_camera2 = [(x - camera2_position[0]) * d_camera2_m / camera2_focal_length for x in x_head_camera2_norm_m]
y_head_camera2 = [(y - camera2_position[1]) * d_camera2_m / camera2_focal_length for y in y_head_camera2_norm_m]
z_head_camera2 = [d_camera2_m for _ in range(len(x_head_camera2_norm_m))]

# Combine head coordinates from both cameras
x_head_combined = [(x_head_camera1[i] + x_head_camera2[i])/2 for i in range(len(x_head_camera1))]
y_head_combined = [(y_head_camera1[i] + y_head_camera2[i])/2 for i in range(len(y_head_camera1))]
z_head_combined = [(z_head_camera1[i] + z_head_camera2[i])/2 for i in range(len(z_head_camera1))]

# Add the coordinates to the head lists
x_head.extend(x_head_combined)
y_head.extend(y_head_combined)
z_head.extend(z_head_combined)


# Calculate the position of the head in 3D space
x_head_3d = []
y_head_3d = []
z_head_3d = []

for i in range(len(x_head)):
    # Convert distances to meters
    d_camera1_m = d_camera1[i] / 100
    d_camera2_m = d_camera2[i] / 100
    
    # Calculate the position of the head in 3D space
    x_head_3d.append((d_camera1_m / math.tan(math.radians(half_angle_of_view))) * ((x_head[i]/x_resolution) - 0.5))
    y_head_3d.append((d_camera1_m / math.tan(math.radians(half_angle_of_view))) * ((y_head[i]/y_resolution) - 0.5))
    z_head_3d.append(d_camera1_m)
    
# Print the 3D position of the head
print("3D Position of the Head:")
for i in range(len(x_head_3d)):
    print(f"Frame {i}: ({x_head_3d[i]:.2f}, {y_head_3d[i]:.2f}, {z_head_3d[i]:.2f})")

# Calculate the position of the ball in 3D space
x_ball = []
y_ball = []
z_ball = []

for i in range(len(x_ball_camera1_norm)):
    # Calculate the distance from each camera to the ball at each frame
    d_camera1_m = d_camera1[i] / 100
    d_camera2_m = d_camera2[i] / 100
    
    # Calculate the position of the ball in 3D space
    x_ball.append((d_camera1_m - d_camera2_m) / math.tan(math.radians(angle_between_cameras)) * (x_ball_camera1_norm[i] - x_ball_camera2_norm[i]))
    y_ball.append((d_camera1_m - d_camera2_m) / math.tan(math.radians(angle_between_cameras)) * (y_ball_camera1_norm[i] - y_ball_camera2_norm[i]))
    z_ball.append(d_camera1_m - ((d_camera1_m - d_camera2_m) / math.tan(math.radians(angle_between_cameras))) * (z_ball_camera1[i] - z_ball_camera2[i]))

# Print the 3D position of the ball
print("3D Position of the Ball:")
for i in range(len(x_ball)):
    print(f"Frame {i}: ({x_ball[i]:.2f}, {y_ball[i]:.2f}, {z_ball[i]:.2f})")

# Calculate the velocity of the ball at each frame
x_ball_velocity = []
y_ball_velocity = []
z_ball_velocity = []

for i in range(1, len(x_ball)):
    x_ball_velocity.append((x_ball[i] - x_ball[i-1]) / time_per_frame)
    y_ball_velocity.append((y_ball[i] - y_ball[i-1]) / time_per_frame)
    z_ball_velocity.append((z_ball[i] - z_ball[i-1]) / time_per_frame)

# Calculate the magnitude of the velocity vector
ball_velocity = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(x_ball_velocity, y_ball_velocity, z_ball_velocity)]

# Print the velocity of the ball
print(f"Ball velocity: {ball_velocity[0]:.2f} m/s")

# Calculate the distance between the ball and the head at each frame
distances = []
for i in range(len(x_ball)):
    distance = math.sqrt((x_head[i] - x_ball[i])**2 + (y_head[i] - y_ball[i])**2 + (z_head[i] - z_ball[i])**2)
    distances.append(distance)

# Find the frame where the ball is closest to the head
min_distance_index = distances.index(min(distances))

# Calculate the deformation of the ball
deformation = 1 - (diameter[min_distance_index] / initial_diameter)

# Print the results
print(f"Deformation: {deformation:.2f}")
print(f"Contact time: {t_contact[min_distance_index]:.2f} s")
print(f"Force exerted by ball on head: {force[min_distance_index]:.2f} N")

