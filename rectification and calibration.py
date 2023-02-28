import numpy as np
import cv2
"""
num_corners: This variable is a tuple that defines the number of inner corners on the chessboard used for calibration. The first value is the number of corners in the horizontal direction, and the second value is the number of corners in the vertical direction.
objpoints_l, objpoints_r, imgpoints_l, and imgpoints_r: These arrays are used to store the object points and image points from all calibration pairs of images. objpoints_l and objpoints_r store the object points for the chessboard, which are the same for all images. imgpoints_l and imgpoints_r store the image points of the chessboard corners in the left and right images, respectively. These arrays are used as inputs to the cv2.stereoCalibrate() function.
objp: This array stores the object points for the chessboard. It is a 3D array with dimensions (num_corners[0]*num_corners[1], 3), where num_corners is the tuple that defines the number of corners on the chessboard. The first two columns of objp are the x and y coordinates of each corner, and the third column is always 0.
ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F: These variables are outputs of the cv2.stereoCalibrate() function, which performs stereo camera calibration. Here's what each variable represents:
ret: A boolean value that indicates whether the calibration was successful.
mtx_l: The camera matrix for the left camera.
dist_l: The distortion coefficients for the left camera.
mtx_r: The camera matrix for the right camera.
dist_r: The distortion coefficients for the right camera.
R: The rotation matrix between the left and right cameras.
T: The translation vector between the left and right cameras.
E: The essential matrix.
F: The fundamental matrix.




R_l, R_r, P_l, P_r, Q, roi_l, roi_r: These variables are outputs of the cv2.stereoRectify() function, which performs stereo rectification. Here's what each variable represents:
R_l: The left camera's 3x3 rectification transform.
R_r: The right camera's 3x3 rectification transform.
P_l: The left camera's 3x4 projection matrix in the new (rectified) coordinate system.
P_r: The right camera's 3x4 projection matrix in the new (rectified) coordinate system.
Q: The disparity-to-depth mapping matrix.
roi_l: The valid pixels in the left camera's rectified image.
roi_r: The valid pixels in the right camera's rectified image.

"""

# Define the number of inner corners on the chessboard
num_corners = (9, 6)

# Create arrays to store object points and image points from all images
objpoints_l = []
objpoints_r = []
imgpoints_l = []
imgpoints_r = []

# Generate the object points for the chessboard
objp = np.zeros((num_corners[0]*num_corners[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:num_corners[0], 0:num_corners[1]].T.reshape(-1,2)

# Loop through each calibration pair of images
for i in range(num_images):
    # Load the left and right images and convert to grayscale
    img_l = cv2.imread(f'calibration_images_left/img{i}.jpg')
    img_r = cv2.imread(f'calibration_images_right/img{i}.jpg')
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners in both images
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, num_corners, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, num_corners, None)

    # If corners are found in both images, add object points and image points
    if ret_l == True and ret_r == True:
        objpoints_l.append(objp)
        objpoints_r.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)

# Calibrate the stereo camera
ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints_l, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1],
    flags=cv2.CALIB_FIX_INTRINSIC)

# Generate the rectification maps
R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1], R, T, alpha=0)

# Save the calibration and rectification parameters to files
np.savez('calibration_params.npz', mtx_l=mtx_l, dist_l=dist_l, mtx_r=mtx_r, dist_r=dist_r, R=R, T=T)
np.savez('rectification_params.npz', R_l=R_l, R_r=R_r, P_l=P_l, P_r=P_r, Q=Q, roi_l=roi_l, roi_r=roi_r)
