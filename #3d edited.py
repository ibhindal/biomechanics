import math

# Define constants
pixels_to_meters = 0.001758
half_angle_of_view = 34.3 / 2
x_resolution = 1920
y_resolution = 1080
diameter = 0.22
initial_diameter = 0.22
t_contact = 0.01
fps = 10000
time_per_frame = 1 / fps
force = 100

# Load data from input files
with open('input_file_camera1.txt', 'r') as f:
    data_camera1 = f.readlines()

with open('input_file_camera2.txt', 'r') as f:
    data_camera2 = f.readlines()

with open('input_file_head_camera1.txt', 'r') as f:
    data_head_camera1 = f.readlines()

with open('input_file_head_camera2.txt', 'r') as f:
    data_head_camera2 = f.readlines()

# Convert xywh to x, y, z for ball in camera 1
x_ball_camera1_norm = []
y_ball_camera1_norm = []
for line in data_camera1:
    x, y, w, h = map(float, line.strip().split())
    x_norm = (x + w/2 - x_resolution/2) / (x_resolution/2)
    y_norm = (y + h/2 - y_resolution/2) / (y_resolution/2)
    x_ball_camera1_norm.append(x_norm)
    y_ball_camera1_norm.append(y_norm)

# Convert xywh to x, y, z for ball in camera 2
x_ball_camera2_norm = []
y_ball_camera2_norm = []
for line in data_camera2:
    x, y, w, h = map(float, line.strip().split())
    x_norm = (x + w/2 - x_resolution/2) / (x_resolution/2)
    y_norm = (y + h/2 - y_resolution/2) / (y_resolution/2)
    x_ball_camera2_norm.append(x_norm)
    y_ball_camera2_norm.append(y_norm)

# Convert xywh to x, y, z for head in camera 1
x_head_camera1_norm = []
y_head_camera1_norm = []
for line in data_head_camera1:
    x, y, w, h = map(float, line.strip().split())
    x_norm = (x + w/2 - x_resolution/2) / (x_resolution/2)
    y_norm = (y + h/2 - y_resolution/2) / (y_resolution/2)
    x_head_camera1_norm.append(x_norm)
    y_head_camera1_norm.append(y_norm)

# Convert xywh to x, y, z for head in camera 2
x_head_camera2_norm = []
y_head_camera2_norm = []
for line in data_head_camera2:
    x, y, w, h = map(float, line.strip().split())
    x_norm = (x + w/2 - x_resolution/2) / (x_resolution/2)
    y_norm = (y + h/2 - y_resolution/2) / (y_resolution/2)
    x_head_camera2_norm.append(x_norm)
    y_head_camera2_norm.append(y_norm)

# Calculate the position of the ball in 3D space
x_ball = []
y_ball = []
z_ball = []
for i in range(len(x_ball_camera1_norm)):
   # Calculate the distance between the camera and the ball
    d_camera1 = pixels_to_meters * diameter / (2 * math.tan(math.radians(half_angle_of_view)) * abs(x_ball_camera1_norm))
    d_camera2 = pixels_to_meters * diameter / (2 * math.tan(math.radians(half_angle_of_view)) * abs(x_ball_camera2_norm))

# Calculate the position of the ball in 3D space
x_ball = []
y_ball = []
z_ball = []
for i in range(len(x_ball_camera1_norm)):
    # Convert distances to meters
    d_camera1_m = d_camera1[i] / 100
    d_camera2_m = d_camera2[i] / 100
    
    # Calculate x, y, z coordinates
    x = (d_camera1_m * x_ball_camera1_norm[i] + d_camera2_m * x_ball_camera2_norm[i]) / 2
    y = (d_camera1_m * y_ball_camera1_norm[i] + d_camera2_m * y_ball_camera2_norm[i]) / 2
    z = (d_camera1_m * z_ball_camera1[i] + d_camera2_m * z_ball_camera2[i]) / 2
    
    # Add the coordinates to the ball lists
    x_ball.append(x)
    y_ball.append(y)
    z_ball.append(z)

# Calculate the velocity of the ball at each frame
x_ball_velocity = []
y_ball_velocity = []
z_ball_velocity = []
for i in range(1, len(x_ball)):
    x_vel = (x_ball[i] - x_ball[i-1]) / time_per_frame
    y_vel = (y_ball[i] - y_ball[i-1]) / time_per_frame
    z_vel = (z_ball[i] - z_ball[i-1]) / time_per_frame
    x_ball_velocity.append(x_vel)
    y_ball_velocity.append(y_vel)
    z_ball_velocity.append(z_vel)

# Calculate the position of the head in 3D space
x_head = []
y_head = []
z_head = []
for i in range(len(x_head_camera1_norm)):
    # Convert distances to meters
    d_camera1_m = d_camera1[i] / 100
    d_camera2_m = d_camera2[i] / 100
    
    # Calculate x, y, z coordinates
    x = (d_camera1_m * x_head_camera1_norm[i] + d_camera2_m * x_head_camera2_norm[i]) / 2
    y = (d_camera1_m * y_head_camera1_norm[i] + d_camera2_m * y_head_camera2_norm[i]) / 2
    z = (d_camera1_m * z_head_camera1[i] + d_camera2_m * z_head_camera2[i]) / 2
    
    # Add the coordinates to the head lists
    x_head.append(x)
    y_head.append(y)
    z_head.append(z)

# Calculate the velocity of the head at each frame
x_head_velocity = []
y_head_velocity = []
z_head_velocity = []
for i in range(1, len(x_head)):
    x_vel = (x_head[i] - x_head[i-1]) / time_per_frame
    y_vel = (y_head[i] - y_head[i-1]) / time_per_frame
    z_vel = (z_head[i] - z_head[i-1]) / time_per_frame
    x_head_velocity.append(x_vel)
    y_head_velocity.append(y_vel)
    z_head_velocity.append(z_vel)

# Calculate the contact time
t_contact = (z_ball[-1] - ball_radius - z_head[-1]) / ball_velocity[-1]


# Calculate the force of the impact
impact_time = t_contact / 2

if impact_time >= time_per_frame and min_distance_idx is not None:
    # Calculate the velocity of the ball at the point of impact
    v_impact = ball_velocity[min_distance_idx - int(impact_time / time_per_frame)]

    # Calculate the force of the impact using the coefficient of restitution
    # and the mass of the ball and player's head
    m_ball = (4/3) * math.pi * ((initial_diameter/2) ** 3) * density
    m_head = player_mass_ratio * body_mass
    v_after = coefficient_of_restitution * v_impact
    delta_v = v_impact - v_after
    impact_force = delta_v * m_ball / impact_time
else:
    impact_force = None




