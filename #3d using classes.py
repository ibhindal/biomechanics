#3d using classes

import math
import math

class Camera:
    def __init__(self, pixels_to_meters, focal_length, half_angle_of_view, x_resolution, y_resolution):
        self.pixels_to_meters = pixels_to_meters
        self.focal_length = focal_length
        self.half_angle_of_view = half_angle_of_view
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution

    def pixel_to_meter(self, pixel):
        return pixel * self.pixels_to_meters

    def meter_to_pixel(self, meter):
        return meter / self.pixels_to_meters

    def distance_to_object(self, pixel_x, pixel_y, diameter):
        distance = diameter * self.pixels_to_meters / (2 * math.tan(math.radians(self.half_angle_of_view)))
        dx = abs(pixel_x - self.x_resolution / 2)
        dy = abs(pixel_y - self.y_resolution / 2)
        object_distance = math.sqrt(distance**2 + dx**2 + dy**2)
        return object_distance

    def velocity_of_object(self, pixel_x1, pixel_y1, pixel_x2, pixel_y2, diameter, time):
        d1 = self.distance_to_object(pixel_x1, pixel_y1, diameter)
        d2 = self.distance_to_object(pixel_x2, pixel_y2, diameter)
        delta_d = d2 - d1
        velocity = delta_d / time
        return velocity


class Ball:
    def __init__(self, pixels_to_meters, diameter, initial_diameter, initial_velocity, initial_position):
        self.pixels_to_meters = pixels_to_meters
        self.diameter = diameter
        self.initial_diameter = initial_diameter
        self.initial_velocity = initial_velocity
        self.initial_position = initial_position

    def radius(self):
        return self.diameter / 2

    def mass(self):
        # assuming density of the ball is 0.006 gm/cm^3
        return 0.006 * (4/3) * math.pi * (self.radius() ** 3)

    def COR(self):
        return 0.7  # assuming a typical value for the coefficient of restitution

    def inbound_velocity(self, outbound_velocity):
        return -1 * self.COR() * outbound_velocity

    def kinetic_energy(self, velocity):
        return 0.5 * self.mass() * (velocity ** 2)

    def velocity_to_position(self, velocity, time):
        delta_x = velocity * time
        return self.initial_position + delta_x


# Define the camera parameters
pixels_to_meters = 0.01
half_angle_of_view = 28.6
x_resolution = 640
y_resolution = 480

# Define the camera objects
camera1 = Camera(pixels_to_meters, 0.05, half_angle_of_view, x_resolution, y_resolution)
camera2 = Camera(pixels_to_meters, 0.05, half_angle_of_view, x_resolution, y_resolution)

# Define the ball parameters
diameter = 0.07
initial_diameter = 0.07
initial_velocity = 0
initial_position = 0

# Define the ball object
ball = Ball(pixels_to_meters, diameter, initial_diameter, initial_velocity, initial_position)

# Define the video parameters
time_between_frames = 0.033  # assuming 30 fps
time_per_frame = 0.033  # assuming 30 fps



class BallTrajectory:
    def __init__(self, pixels_to_meters, diameter, initial_diameter, x_resolution, y_resolution, fps, half_angle_of_view):
        self.pixels_to_meters = pixels_to_meters
        self.diameter = diameter
        self.initial_diameter = initial_diameter
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.fps = fps
        self.half_angle_of_view = half_angle_of_view
        self.ball_positions = []
        self.head_positions = []
        self.t_contact = None
        self.ball_velocity_in = None
        self.ball_velocity_out = None
        self.force = None

    def add_ball_position(self, x, y, w, h, camera):
        x_norm = x / self.x_resolution - 0.5
        y_norm = y / self.y_resolution - 0.5
        x_meters = x_norm * self.pixels_to_meters * camera.distance
        y_meters = y_norm * self.pixels_to_meters * camera.distance
        z_meters = math.sqrt(camera.distance ** 2 - x_meters ** 2 - y_meters ** 2)
        self.ball_positions.append((x_meters, y_meters, z_meters))

    def add_head_position(self, x, y, w, h, camera):
        x_norm = x / self.x_resolution - 0.5
        y_norm = y / self.y_resolution - 0.5
        x_meters = x_norm * self.pixels_to_meters * camera.distance
        y_meters = y_norm * self.pixels_to_meters * camera.distance
        z_meters = math.sqrt(camera.distance ** 2 - x_meters ** 2 - y_meters ** 2)
        self.head_positions.append((x_meters, y_meters, z_meters))

    def calculate_contact_time(self):
        z_head_combined = [head_pos[2] for head_pos in self.head_positions]
        avg_z_head = sum(z_head_combined) / len(z_head_combined)
        avg_d = avg_z_head - self.ball_positions[-1][2]
        self.t_contact = (self.diameter * math.sqrt(self.initial_diameter ** 2 - self.diameter ** 2)) / (self.ball_velocity_in * avg_d)

    def calculate_ball_velocity(self):
        x_ball_combined = [ball_pos[0] for ball_pos in self.ball_positions]
        y_ball_combined = [ball_pos[1] for ball_pos in self.ball_positions]
        z_ball_combined = [ball_pos[2] for ball_pos in self.ball_positions]
        x_head_combined = [head_pos[0] for head_pos in self.head_positions]
        y_head_combined = [head_pos[1] for head_pos in self.head_positions]
        z_head_combined = [head_pos[2] for head_pos in self.head_positions]

        d_camera1 = self.pixels_to_meters * self.diameter / (2 * math.tan(math.radians(self.half_angle_of_view)) * abs(x_ball_combined[-1] - x_head_combined[-1]))
        d_camera2 = self.pixels_to_meters * self.diameter / (2 * math.tan(math.radians(self.half_angle_of_view)) * abs(x_ball_combined[-1] - x_head_combined[-1] - 0.5))

        x_ball_camera1_norm = [(x - camera.x_resolution / 2) / camera.x_resolution for (x, _, _) in self.ball_positions]
        y_ball_camera1_norm = [(y - camera.y_resolution / 2) / camera.y_resolution for (_, y, _) in self.ball_positions]
        # Normalize the x and y positions of the ball in camera 2
    x_ball_camera2_norm = [(x - camera.x_resolution / 2) / camera.x_resolution for (x, _, _) in self.ball_positions]
    y_ball_camera2_norm = [(y - camera.y_resolution / 2) / camera.y_resolution for (_, y, _) in self.ball_positions]

    # Convert the diameter of the ball from pixels to meters
    diameter_m = self.diameter / camera.pixels_to_meters

    # Calculate the distance between the cameras
    distance_between_cameras = camera.distance_between_cameras

    # Calculate the distance between the ball and camera 1
    d_camera1 = [camera.pixels_to_meters * diameter_m / (2 * math.tan(math.radians(camera.half_angle_of_view)) * abs(x))
                for x in x_ball_camera1_norm]

    # Calculate the distance between the ball and camera 2
    d_camera2 = [camera.pixels_to_meters * diameter_m / (2 * math.tan(math.radians(camera.half_angle_of_view)) * abs(x))
                for x in x_ball_camera2_norm]

    # Calculate the 3D position of the ball in camera 1
    x_ball_camera1 = [d * x for (d, x) in zip(d_camera1, x_ball_camera1_norm)]
    y_ball_camera1 = [d * y for (d, y) in zip(d_camera1, y_ball_camera1_norm)]
    z_ball_camera1 = [math.sqrt(d ** 2 - x ** 2 - y ** 2) for (d, x, y) in zip(d_camera1, x_ball_camera1_norm, y_ball_camera1_norm)]

    # Calculate the 3D position of the ball in camera 2
    x_ball_camera2 = [distance_between_cameras - d for d in d_camera2]
    y_ball_camera2 = [d * y for (d, y) in zip(d_camera2, y_ball_camera2_norm)]
    z_ball_camera2 = [math.sqrt(d ** 2 - x ** 2 - y ** 2) for (d, x, y) in zip(d_camera2, x_ball_camera2_norm, y_ball_camera2_norm)]

    # Calculate the 3D position of the ball in space using triangulation
    x_ball = [(d1 ** 2 - d2 ** 2 + distance_between_cameras ** 2) / (2 * distance_between_cameras) for (d1, d2) in zip(d_camera1, d_camera2)]
    y_ball = [(y1 * d2 - y2 * d1) / math.sqrt(d1 ** 2 - x ** 2 - y ** 2) for (d1, d2, x, y, y1, y2) in zip(d_camera1, d_camera2, x_ball, y_ball, y_ball_camera1, y_ball_camera2)]
    z_ball = [math.sqrt(d1 ** 2 - x ** 2 - y ** 2) for (d1, x, y) in zip(d_camera1, x_ball, y_ball)]

    # Calculate the position of the head in camera 1
    x_head_camera1_norm = [(x + camera.head_offset_x) / camera.x_resolution for x in self.head_positions]
    y_head_camera1_norm = [(y + camera.head_offset_y) / camera.y_resolution for y in self.head_positions]
    z_head_camera1_norm = [camera.d_camera_head / math.tan(math.radians(camera.head_angle_of_view)) for _ in range(len(self.head_positions))]

    # Calculate the position of the head in camera 2
    x_head_camera2_norm = [(x + camera.head_offset_x) / camera.x_resolution for x in self.head_positions]
    y_head_camera2_norm = [(y + camera.head_offset_y) / camera.y_resolution for y in self.head_positions]


    #Calculate the position of the head in camera 2
    x_head_camera2_norm = [(x - camera.x_resolution / 2 - camera.distance_between_cameras / 2) / camera.x_resolution for (x, _, _) in self.head_positions]
    y_head_camera2_norm = y_head_camera1_norm

    #Convert normalized camera coordinates to meters
    x_head_camera1_m = [x * camera.d_camera_head * math.tan(math.radians(camera.head_angle_of_view)) for x in x_head_camera1_norm]
    y_head_camera1_m = [y * camera.d_camera_head * math.tan(math.radians(camera.head_angle_of_view)) for y in y_head_camera1_norm]
    z_head_camera1_m = [camera.d_camera_head for _ in range(len(self.head_positions))]

    x_head_camera2_m = [x * camera.d_camera_head * math.tan(math.radians(camera.head_angle_of_view)) for x in x_head_camera2_norm]
    y_head_camera2_m = [y * camera.d_camera_head * math.tan(math.radians(camera.head_angle_of_view)) for y in y_head_camera2_norm]
    z_head_camera2_m = [camera.d_camera_head for _ in range(len(self.head_positions))]

    #Combine the camera 1 and camera 2 head positions
    x_head_combined = [(x_head_camera1_m[i] + x_head_camera2_m[i] + camera.distance_between_cameras / 2) / 1000 for i in range(len(x_head_camera1_m))]
    y_head_combined = [(y_head_camera1_m[i] + y_head_camera2_m[i]) / 1000 for i in range(len(y_head_camera1_m))]
    z_head_combined = [2 * camera.d_camera_head - (z_head_camera1_m[i] + z_head_camera2_m[i]) / 1000 for i in range(len(z_head_camera1_m))]

    #Add the coordinates to the head lists
    self.x_head.extend(x_head_combined)
    self.y_head.extend(y_head_combined)
    self.z_head.extend(z_head_combined)

   # Calculate the position of the ball in 3D space
    x_ball = []
    y_ball = []
    z_ball = []

    for i in range(len(x_ball_camera1_norm)):
    # Calculate the distance between the camera and the ball
        d_camera1 = self.pixels_to_meters * self.diameter / (2 * math.tan(math.radians(self.half_angle_of_view)) * abs(x_ball_camera1_norm[i]))
        d_camera2 = self.pixels_to_meters * self.diameter / (2 * math.tan(math.radians(self.half_angle_of_view)) * abs(x_ball_camera2_norm[i]))

        # Calculate the coordinates of the ball in 3D space
    x_ball_camera1 = d_camera1 * x_ball_camera1_norm[i]
    y_ball_camera1 = d_camera1 * y_ball_camera1_norm[i]
    z_ball_camera1 = d_camera1 * math.sqrt(1 - x_ball_camera1_norm[i] ** 2 - y_ball_camera1_norm[i] ** 2)

    x_ball_camera2 = d_camera2 * x_ball_camera2_norm[i]
    y_ball_camera2 = d_camera2 * y_ball_camera2_norm[i]
    z_ball_camera2 = d_camera2 * math.sqrt(1 - x_ball_camera2_norm[i] ** 2 - y_ball_camera2_norm[i] ** 2)

    x_ball_combined = (x_ball_camera1 + x_ball_camera2) / 2
    y_ball_combined = (y_ball_camera1 + y_ball_camera2) / 2
    z_ball_combined = (z_ball_camera1 + z_ball_camera2) / 2

    x_ball.append(x_ball_combined)
    y_ball.append(y_ball_combined)
    z_ball.append(z_ball_combined)

    #Calculate the position of the head in 3D space
    x_head = []
    y_head = []
    z_head = []
    for i in range(len(x_head_camera1_norm)):
    # Convert distances to meters
        d_camera1_m = self.d_camera_head / math.tan(math.radians(self.head_angle_of_view))
        d_camera2_m = self.d_camera_head / math.tan(math.radians(self.head_angle_of_view))
    
    # Calculate the coordinates of the head in 3D space
    x_head_camera1_m = d_camera1_m * x_head_camera1_norm[i]
    y_head_camera1_m = d_camera1_m * y_head_camera1_norm[i]
    z_head_camera1_m = d_camera1_m * math.sqrt(1 - x_head_camera1_norm[i] ** 2 - y_head_camera1_norm[i] ** 2)

    x_head_camera2_m = d_camera2_m * x_head_camera2_norm[i]
    y_head_camera2_m = d_camera2_m * y_head_camera2_norm[i]
    z_head_camera2_m = d_camera2_m * math.sqrt(1 - x_head_camera2_norm[i] ** 2 - y_head_camera2_norm[i] ** 2)

    x_head_combined = (x_head_camera1_m + x_head_camera2_m) / 2
    y_head_combined = (y_head_camera1_m + y_head_camera2_m) / 2
    z_head_combined = (z_head_camera1_m + z_head_camera2_m) / 2

    x_head.append(x_head_combined)
    y_head.append(y_head_combined)
    z_head.append(z_head_combined)

   # Calculate the velocity of the ball
    ball_velocity = [0, 0, 0]
    if len(x_ball) > 1:
        ball_velocity = [(x_ball[-1] - x_ball[0]) / (self.time_per_frame * len(x_ball)), (y_ball[-1] - y_ball[0]) / (self.time_per_frame * len(y_ball)), (z_ball[-1] - z_ball[0]) / (self.time_per_frame * len(z_ball))]


#Calculate the contact time
t_contact = (z_ball_camera1[-1] - z_head_camera1[-1]) / ball_velocity[0]

#Calculate the COR using the formula COR = |v_out| / |v_in|
v_in = ball_velocity[0]
v_out = ball_velocity[-1]
COR = abs(v_out) / abs(v_in)

#Calculate the impulse using the formula impulse = F * t_contact
impulse = force * t_contact

#Print the results
print(f"Inbound velocity: {v_in:.2f} m/s")
print(f"Outbound velocity: {v_out:.2f} m/s")
print(f"COR: {COR:.2f}")
print(f"Impulse: {impulse:.2f} Ns")
