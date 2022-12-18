import cv2
import numpy as np
import os
import pandas as pd
import scipy
import math
#os.chdir("C:\\Users\\Ibrahim\\desktop")

tracker = cv2.TrackerCSRT_create()
video = cv2.VideoCapture('CL_10_S0001.mp4')
ok,frame=video.read()
bbox = cv2.selectROI(frame)
ok = tracker.init(frame,bbox)

wall_bbox = cv2.selectROI(frame)
(x_wall,y_wall,x2_wall,y2_wall) = wall_bbox
num_cont_frames=0

#video = cv.VideoCapture(path)
ball_size = 0.22 #diameter of a regulation ball in meters
fps_cam = 10000 # Change this to the required fps of the video 
fps_vid =video.get(cv2.CAP_PROP_FPS)
fps_time= fps_vid / fps_cam 
#print(fps_time)

scale  = []
x_list = []
y_list = []
x_def=[]


while True:
    ok,frame=video.read()
    if not ok:
        break
    ok,bbox=tracker.update(frame)
    if ok:

        
        (x,y,w,h)=[int(v) for v in bbox]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2,1)
        x2=x+w
        y2=y+h
        scale.append(ball_size/h)  #meters per pixel.diameter in pixels or coordinate value / real diameter in m to give pixel per m for a scale factor  
        x_list.append(x2) #list of x positions of right edge
        y_list.append(y2) 
        if x < x2_wall and y < y2_wall:
            # Increment counter
            num_cont_frames += 1
            x_defe = x2-x2_wall
            x_def.append(x_defe) 
        print(num_cont_frames)
    
    else:
        cv2.putText(frame,'Error',(100,0),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow('Tracking',frame)
    if cv2.waitKey(1) & 0XFF==27:
        break
    
cv2.destroyAllWindows()

scale_ave=scipy.stats.trim_mean(scale, 0.2) #trim_mean 20% either way to remove some extrainious results

x_diff=[]
y_diff=[]
x_len=len(x_list)-1 #minus 1 as python starts with 0 so we dont overflow

for i in range(x_len): 
        x_diff.append(x_list[i]-x_list[i+1]) #find x distance per frame


for i in range(x_len): 
    y_diff.append(y_list[i]-y_list[i+1])  #find y distance per frame


pyth_dist=[]
pyth_sub=[]
x2_len=len(x_diff)-1

for i in range(x2_len):
    pyth_sub=math.hypot(x_diff[i] , y_diff[i])
    pyth_dist.append(pyth_sub) #do pythagoras to find pixel distance per frame
  
realdist=[]
speed=[]
for i in range(x2_len):
    realdistcalc=(pyth_dist[i]*scale_ave)
    realdist.append(realdistcalc) # change from pixels to meters

for item in realdist:
    if item > 1:
        realdist.remove(item)

distlen=len(realdist)-1


for i in range(distlen):
    speedcalc=realdist[i]/fps_time
    speed.append(speedcalc)

contact_time=num_cont_frames/fps_cam
print(contact_time)

if x_def:
    realxdef = min(x_def)*scale_ave
else:
    realxdef = 0

print(realxdef)

df = pd.DataFrame(speed)
df.to_csv('speed.csv', index=False)