#2d heading

import math

# Define constants
ball_mass = 0.45  # kg
ball_radius = 0.11  # meters
g = 9.81  # m/s^2
frame_rate = 10000  # frames per second
ball_diameter = 0.22  # meters

# Define variables
x_ball = []
y_ball = []
x_head = []
y_head = []
t = []
t_contact = None

# Read in data from file or user input
# Append the values of x_ball, y_ball, x_head, y_head, and t accordingly for each frame of the video

# Calculate the inbound and outbound speeds
x_ball_velocity = (x_ball[1] - x_ball[0]) * frame_rate
y_ball_velocity = (y_ball[1] - y_ball[0]) * frame_rate

# Calculate the velocity of the ball's center of mass
ball_velocity = math.sqrt(x_ball_velocity**2 + y_ball_velocity**2)

# Calculate the deformation of the ball
ball_diameter_deformed = ball_diameter * (1 - deformation)
ball_radius_deformed = ball_diameter_deformed / 2
y_ball_head = y_ball[-1] - y_head[-1]
if y_ball_head > ball_radius_deformed:
    y_ball_deformed = y_head[-1] + ball_radius_deformed
else:
    y_ball_deformed = y_head[-1] + math.sqrt(ball_radius_deformed**2 - y_ball_head**2)
deformation = (ball_radius - ball_radius_deformed) / ball_radius

# Calculate the contact time
contact_start = 0
for i in range(1, len(t)):
    if y_ball[i] > y_head[i] and y_ball[i-1] <= y_head[i-1]:
        contact_start = t[i]
    if y_ball[i] <= y_head[i] and y_ball[i-1] > y_head[i-1]:
        t_contact = t[i] - contact_start
        break

# Calculate the force exerted by the ball on the head
impulse = ball_mass * ball_velocity
force = impulse / t_contact

# Print the results
print(f"Inbound speed: {x_ball_velocity:.2f} m/s")
print(f"Outbound speed: {math.sqrt(x_ball_velocity**2 + (y_ball_velocity - 2*g*t_contact)**2):.2f} m/s")
print(f"Deformation: {deformation:.2f}")
print(f"Contact time: {t_contact:.2f} s")
print(f"Force exerted by ball on head: {force:.2f} N")
