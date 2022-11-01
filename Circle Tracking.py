# Hough circle transform technique
#https://www.youtube.com/watch?v=RaCwLrKuS1w


import cv2 as cv
import numpy as np
import os #provides functions for interacting with operating system (OS)
import math
#os.system('cls')
import pandas as pd

# os.chdir : changes the current working directory to specified path
# Syntax: os.chdir(path)
os.chdir("C:\\Users\\Ibrahim\\biomechanics")




name = 'P2_2_30_1.mp4' # white ball : P2_1_30_1.mp4 | orange ball: P2_2_30_1.mp4
video = cv.VideoCapture(name)  
prevCircle = None
dist = lambda x1, y1, x2, y2: (x1-x2)**2 + (y1-y2)**2

#video = cv.VideoCapture(path)
ball_size = 22 #diameter of a regulation ball
fps_cam = 1500 # Change this to the required fps of the video 
fps_vid =video.get(cv.CAP_PROP_FPS)
fps_time= fps_vid / fps_cam
print(fps_time)

scale=[]
x_list=[]
y_list=[]

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
        #print(chosen) #these print statements are for debugging andhelp with understanding the code.
        #they do fill up the terminal with information tho

        ball_d= chosen[2]
        #print(ball_d)
        scale.append(ball_d/ball_size)  #diameter in pixels or coordinate value / real diameter in cm to give pixel per cm for a scale factor  
        x_list.append(chosen[0])
        y_list.append(chosen[1])
        
        #print(scale)
        

    cv.imshow("circles", frame)
    #print(float(prevCircle[dist]) - float(chosen[dist]))


    if cv.waitKey(1) & 0xFF == ord('q'): break


video.release()
cv.destroyAllWindows()


x_diff=[]
x_len=len(x_list)-1 #minus 1 as python starts with 0 so we dont overflow

for i in range(x_len): 
        x_diff.append(x_list[i]-x_list[i+1])
#print(x_diff)

y_diff=[]

for i in range(x_len): 
    y_diff.append(y_list[i]-y_list[i+1])
#print(y_diff)

pyth_dist=[]
pyth_sub=[]
x2_len=len(x_diff)-1

for i in range(x2_len):
    pyth_sub=math.hypot(x_diff[i] , y_diff[i])
    pyth_dist.append(pyth_sub)

#print(pyth_dist)
df = pd.DataFrame(pyth_dist)
df.to_csv('pyth_dist.csv', index=False)
