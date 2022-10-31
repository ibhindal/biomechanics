# Hough circle transform technique
#https://www.youtube.com/watch?v=RaCwLrKuS1w


import cv2 as cv
import numpy as np
import os #provides functions for interacting with operating system (OS)

#os.system('cls')


# os.chdir : changes the current working directory to specified path
# Syntax: os.chdir(path)

os.chdir("C:\\Users\\Ibrahim\\biomechanics")

video = cv.VideoCapture('P2_2_30_1.mp4')  # white ball : P2_1_30_1.mp4 | orange ball: P2_2_30_1.mp4
prevCircle = None
dist = lambda x1, y1, x2, y2: (x1-x2)**2 + (y1-y2)**2
#video = cv.VideoCapture(path)

fps_cam = 1500 # Change this to the required fps of the video 
fps_vid =video.get(cv.CAP_PROP_FPS)
fps_time= fps_vid / fps_cam
print(fps_time)




while True:    

    #read (returns) frame from video capture
    ret, frame = video.read()

    # if frame not returned, subsequent break function is performed
    if not ret: break

#converts frame to grayscale and blurs it. (Helps mitigate background noise)

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #17,17) input decides degree of blur. Higher integers increase blur
    blurFrame = cv.GaussianBlur(grayFrame, (17,17), 0)
    
    #Hough Circle Transform :

    circles = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1.2, 100, 
                param1 = 100, param2 = 30, minRadius = 300, maxRadius= 350)  # orange ball

# goes through detected circles and selects the "best" one

    if circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(i[0], i[1],prevCircle[0], prevCircle[1]):
                    chosen = i

#drawing circle to detect centre point of object and around the ball itself
# Where 3 represents thickness of circle
        cv.circle(frame, (chosen[0], chosen[1]), 1, (0,100,100), 3)

#circle around ball or circumference of detected circle

        cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (255,0,255), 3)
        prevCircle = chosen


    cv.imshow("circles", frame)
    #print(float(prevCircle[dist]) - float(chosen[dist]))




    if cv.waitKey(1) & 0xFF == ord('q'): break

    

video.release()
cv.destroyAllWindows()


