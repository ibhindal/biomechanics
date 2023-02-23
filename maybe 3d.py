import math

# Camera positions
camera1_position = [0, 0, 0]
camera2_position = [0.5, 0, 0]


# Camera angles
angle_between_cameras = 10

# Ball diameter
diameter = 0.22

# Ball mass
mass = 0.45

# Acceleration due to gravity
g = 9.81

# Load data from input files
with open("ball_camera1.txt", "r") as f:
    ball_camera1_data = f.readlines()

with open("ball_camera2.txt", "r") as f:
    ball_camera2_data = f.readlines()

with open("head_camera1.txt", "r") as f:
    head_camera1_data = f.readlines()

with open("head_camera2.txt", "r") as f:
    head_camera2_data = f.readlines()

# Parse data from input files
x_ball_camera1 = [int(line.split()[0]) for line in ball_camera1_data]
y_ball_camera1 = [int(line.split()[1]) for line in ball_camera1_data]
w_ball_camera1 = [int(line.split()[2]) for line in ball_camera1_data]
h_ball_camera1 = [int(line.split()[3]) for line in ball_camera1_data]

x_ball_camera2 = [int(line.split()[0]) for line in ball_camera2_data]
y_ball_camera2 = [int(line.split()[1]) for line in ball_camera2_data]
w_ball_camera2 = [int(line.split()[2]) for line in ball_camera2_data]
h_ball_camera2 = [int(line.split()[3]) for line in ball_camera2_data]

x_head_camera1 = [int(line.split()[0]) for line in head_camera1_data]
y_head_camera1 = [int(line.split()[1]) for line in head_camera1_data]
w_head_camera1 = [int(line.split()[2]) for line in head_camera1_data]
h_head_camera1 = [int(line.split()[3]) for line in head_camera1_data]

x_head_camera2 = [int(line.split()[0]) for line in head_camera2_data]
y_head_camera2 = [int(line.split()[1]) for line in head_camera2_data]
w_head_camera2 = [int(line.split()[2]) for line in head_camera2_data]
h_head_camera2 = [int(line.split()[3]) for line in head_camera2_data]

# Convert xywh to x, y, z for ball in camera 1
x_ball_camera1_norm = [(x + w/2) / 1920 for x, w in zip(x_ball_camera1, w_ball_camera1)]
y_ball_camera1_norm = [(y + h/2) / 1080 for y, h in zip(y_ball_camera1, h_ball_camera1)]
z_ball_camera1 = [d_camera1/100 for d_camera1 in w_ball_camera1]

# Convert xywh to x, y, z for ball in camera 2
x_ball_camera2_norm = [(x + w/2) / 1920 for x, w in zip(x_ball_camera2, w_ball_camera2)]
y_ball_camera2_norm = [(y + h/2) / 1080 for y, h in zip(y_ball_camera2, h_ball_camera2)]
z_ball_camera2 = [d_camera2/100 for d_camera2 in w_ball_camera2]

#Convert xywh to x, y, z for head in camera 1
x_head_camera1_norm = [(x + w/2)/width_camera1 for (x, y, w, h) in head_camera1_xywh]
y_head_camera1_norm = [(y + h/2)/height_camera1 for (x, y, w, h) in head_camera1_xywh]
z_head_camera1 = [camera1_position[2] - (head_height/2 + y*head_height) for y in y_head_camera1_norm]

#Convert xywh to x, y, z for head in camera 2
x_head_camera2_norm = [(x + w/2)/width_camera2 for (x, y, w, h) in head_camera2_xywh]
y_head_camera2_norm = [(y + h/2)/height_camera2 for (x, y, w, h) in head_camera2_xywh]
z_head_camera2 = [camera2_position[2] - (head_height/2 + y*head_height) for y in y_head_camera2_norm]

#Convert xywh to x, y for ball in camera 1
x_ball_camera1_norm = [(x + w/2)/width_camera1 for (x, y, w, h) in ball_camera1_xywh]
y_ball_camera1_norm = [(y + h/2)/height_camera1 for (x, y, w, h) in ball_camera1_xywh]

#Convert xywh to x, y for ball in camera 2
x_ball_camera2_norm = [(x + w/2)/width_camera2 for (x, y, w, h) in ball_camera2_xywh]
y_ball_camera2_norm = [(y + h/2)/height_camera2 for (x, y, w, h) in ball_camera2_xywh]

#Calculate the position of the head in 3D space
#First, calculate the distance from each camera to the head at each frame
d_head_camera1 = [math.sqrt((x_head_camera1_norm[i] - camera1_position[0])**2 + (y_head_camera1_norm[i] - camera1_position[1])**2) for i in range(num_frames)]
d_head_camera2 = [math.sqrt((x_head_camera2_norm[i] - camera2_position[0])**2 + (y_head_camera2_norm[i] - camera2_position[1])**2) for i in range(num_frames)]

#Next, calculate the x, y, and z coordinates of the head in 3D space
x_head = [(camera1_position[2] - camera2_position[2]) / math.tan(math.radians(angle_between_cameras)) * (x_head_camera1_norm[i] - x_head_camera2_norm[i]) for i in range(num_frames)]
y_head = [(camera1_position[2] - camera2_position[2]) / math.tan(math.radians(angle_between_cameras)) * (y_head_camera1_norm[i] - y_head_camera2_norm[i]) for i in range(num_frames)]
z_head = [d_head_camera1[i] * math.sin(math.radians(angle_camera1_head)) for i in range(num_frames)]

# Calculate the position of the ball in 3D space
# First, calculate the distance from each camera to the ball at each frame
d_camera1 = [math.sqrt((x_ball_camera1_norm[i] - camera1_position[0])**2 + (y_ball_camera1_norm[i] - camera1_position[1])**2) for i in range(len(x_ball_camera1_norm))]
d_camera2 = [math.sqrt((x_ball_camera2_norm[i] - camera2_position[0])**2 + (y_ball_camera2_norm[i] - camera2_position[1])**2) for i in range(len(x_ball_camera2_norm))]

# Next, calculate the x, y, and z coordinates of the ball in 3D space
x_ball = []
y_ball = []
z_ball = []
for i in range(len(d_camera1)):
    x_ball.append((camera1_position[2] - camera2_position[2]) / math.tan(math.radians(angle_between_cameras)) * (x_ball_camera1_norm[i] - x_ball_camera2_norm[i]))
    y_ball.append((camera1_position[2] - camera2_position[2]) / math.tan(math.radians(angle_between_cameras)) * (y_ball_camera1_norm[i] - y_ball_camera2_norm[i]))
    z_ball.append((camera1_position[2] - camera2_position[2]) / math.tan(math.radians(angle_between_cameras)) * (d_camera1[i] + d_camera2[i]) / 2)

# Calculate the velocity of the ball at each frame
x_ball_velocity = []
y_ball_velocity = []
z_ball_velocity = []
for i in range(1, len(x_ball)):
    x_ball_velocity.append((x_ball[i] - x_ball[i-1]) * fps)
    y_ball_velocity.append((y_ball[i] - y_ball[i-1]) * fps)
    z_ball_velocity.append((z_ball[i] - z_ball[i-1]) * fps)

# Calculate the speed of the ball at the moment of impact
impact_frame = x_ball_velocity.index(max(x_ball_velocity))
x_ball_velocity_at_impact = x_ball_velocity[impact_frame]
y_ball_velocity_at_impact = y_ball_velocity[impact_frame]
z_ball_velocity_at_impact = z_ball_velocity[impact_frame]
impact_speed = math.sqrt(x_ball_velocity_at_impact**2 + y_ball_velocity_at_impact**2 + z_ball_velocity_at_impact**2)

# Convert xywh to x, y, z for head in camera 1
x_head_camera1_norm = [(x + w/2) / camera1_resolution[0] for x, y, w, h in head_camera1]
y_head_camera1_norm = [(y + h/2) / camera1_resolution[1] for x, y, w, h in head_camera1]
z_head_camera1 = [camera1_position[2] for _ in range(len(head_camera1))]

# Convert xywh to x, y, z for head in camera 2
x_head_camera2_norm = [(x + w/2) / camera2_resolution[0] for x, y, w, h in head_camera2]
y_head_camera2_norm = [(y + h/2) / camera2_resolution[1] for x, y, w, h in head_camera2]
z_head_camera2 = [camera2_position[2] for _ in range(len(head_camera2))]

# Calculate the position of the head in 3D space
Calculate the position of the head in 3D space
x_head = []
y_head = []
z_head = []
for i in range(len(x_head_camera1_norm)):
# Calculate the distance from each camera to the head at each frame
    d_camera1 = math.sqrt((x_head_camera1_norm[i] - camera1_position[0])**2 + (y_head_camera1_norm[i] - camera1_position[1])**2)
    d_camera2 = math.sqrt((x_head_camera2_norm[i] - camera2_position[0])**2 + (y_head_camera2_norm[i] - camera2_position[1])**2)


# Calculate the x, y, and z positions of the head in 3D space
x_head.append((camera1_position[2] - camera2_position[2]) / math.tan(math.radians(angle_between_cameras)) * (x_head_camera1_norm[i] - x_head_camera2_norm[i]))
y_head.append((camera1_position[2] - camera2_position[2]) / math.tan(math.radians(angle_between_cameras)) * (y_head_camera1_norm[i] - y_head_camera2_norm[i]))
z_head.append((camera1_position[2] - camera2_position[2]) / math.tan(math.radians(angle_between_cameras)) * (z_head_camera1[i] - z_head_camera2[i]))
#Calculate the distance between the ball and head at each frame
distance = [math.sqrt((x_ball[i] - x_head[i])**2 + (y_ball[i] - y_head[i])**2 + (z_ball[i] - z_head[i])**2) for i in range(len(x_ball))]

#Find the frame where the distance is smallest
min_distance_frame = distance.index(min(distance))

#Calculate the deformation of the ball at the point of contact
initial_diameter = 22 # cm
final_diameter = initial_diameter * min_distance / min(distance)
deformation = initial_diameter - final_diameter

#Calculate the velocity of the ball at the point of contact
t_contact = min_distance_frame / fps
x_ball_velocity_contact = (x_ball[min_distance_frame] - x_ball[min_distance_frame-1]) / (1/fps)
y_ball_velocity_contact = (y_ball[min_distance_frame] - y_ball[min_distance_frame-1]) / (1/fps)
z_ball_velocity_contact = (z_ball[min_distance_frame] - z_ball[min_distance_frame-1]) / (1/fps)

#Calculate the inbound and outbound speed of the ball
x_ball_velocity = x_ball_velocity_contact / math.cos(math.atan(y_ball_velocity_contact / x_ball_velocity_contact))
y_ball_velocity = y_ball_velocity_contact / math.cos(math.atan(x_ball_velocity_contact / y_ball_velocity_contact))
inbound_speed = math.sqrt(x_ball_velocity2 + y_ball_velocity2)
outbound_speed = math.sqrt(x_ball_velocity**2 + (y_ball_velocity - 2gt_contact)**2)

#Calculate the force exerted by the ball on the head
mass = ball_mass
force = mass * (outbound_speed - inbound_speed) / t_contact


#Print the results
print(f"Inbound speed: {inbound_speed:.2f} m/s")
print(f"Outbound speed: {outbound_speed:.2f} m/s")
print(f"Deformation: {deformation:.2f}")
print(f"Contact time: {t_contact:.2f} s")
print(f"Force exerted by ball on head: {force:.2f} N")