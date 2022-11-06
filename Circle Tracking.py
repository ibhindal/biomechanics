# Hough circle transform technique
#https://www.youtube.com/watch?v=RaCwLrKuS1w


import cv2 as cv
import numpy as np
import os #provides functions for interacting with operating system (OS)
import math
import scipy
#os.system('cls')
import pandas as pd
import telegram_send

# os.chdir : changes the current working directory to specified path
# Syntax: os.chdir(path)
os.chdir("C:\\Users\\Ibrahim\\biomechanics")

file_name=[]

for x in range(1, 3, 1 ):
 for y in range(30, 60, 10):
    for z in range(1, 4, 1 ):
        name =("P2_" + str(x) +"_" +str(y) +"_" + str(z)+ ".mp4")
        #print (name)
        file_name.append(name) 

#print (file_name)

file_name_len=len(file_name)
for i in range(file_name_len): 
    name = file_name[i]
   
    #name = 'P2_2_30_1.mp4' # white ball : P2_1_30_1.mp4 | orange ball: P2_2_30_1.mp4
    video = cv.VideoCapture(name)  
    prevCircle = None
    dist = lambda x1, y1, x2, y2: (x1-x2)**2 + (y1-y2)**2

    #video = cv.VideoCapture(path)
    ball_size = 0.22 #diameter of a regulation ball in meters
    fps_cam = 1500 # Change this to the required fps of the video 
    fps_vid =video.get(cv.CAP_PROP_FPS)
    fps_time= fps_vid / fps_cam 
    #print(fps_time)

    scale  = []
    x_list = []
    y_list = []

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

            ball_d= chosen[2]*2 #changes radius to diameter of ball
            #print(ball_d)
            scale.append(ball_d/ball_size)  #diameter in pixels or coordinate value / real diameter in cm to give pixel per cm for a scale factor  
            x_list.append(chosen[0]) #list of x positions of center of ball
            y_list.append(chosen[1]) # list of y positions of center of ball
            
            print(scale)
            

        cv.imshow("circles", frame)
        #print(float(prevCircle[dist]) - float(chosen[dist]))


        if cv.waitKey(1) & 0xFF == ord('q'): break


    video.release()
    cv.destroyAllWindows()

    scale_ave=scipy.stats.trim_mean(scale, 0.1) #trim_mean 10% either way to remove some extrainious results
    
    x_diff=[]
    x_len=len(x_list)-1 #minus 1 as python starts with 0 so we dont overflow

    for i in range(x_len): 
            x_diff.append(x_list[i]-x_list[i+1]) #find x distance per frame
    #print(x_diff)

    y_diff=[]

    for i in range(x_len): 
        y_diff.append(y_list[i]-y_list[i+1])  #find y distance per frame
    #print(y_diff)

    pyth_dist=[]
    pyth_sub=[]
    x2_len=len(x_diff)-1

    for i in range(x2_len):
        pyth_sub=math.hypot(x_diff[i] , y_diff[i])
        pyth_dist.append(pyth_sub) #do pythagoras to find pixel distance per frame
        #print(pyth_dist)



    realdist=[]
    speed=[]
    for i in range(x2_len):
        realdistcalc=(pyth_dist[i]/scale_ave)
        realdist.append(realdistcalc) # chande from pixels to meters

    #print(realdist)

    for item in realdist:
        if item > 1:
            realdist.remove(item)

    distlen=len(realdist)-1
    
    for i in range(distlen):
        speedcalc=realdist[i]/fps_time
        speed.append(speedcalc)


   # print(speed)
    df = pd.DataFrame(speed)
    df.to_csv('speed.csv', index=False)