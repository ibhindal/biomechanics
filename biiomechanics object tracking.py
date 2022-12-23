import cv2
import numpy as np
import os
import pandas as pd
import scipy
import math
#os.chdir("C:\\Users\\Ibrahim\\desktop")
file='CL_10_S0001.mp4'

tracker = cv2.TrackerCSRT_create()
video = cv2.VideoCapture(file)
ok,frame=video.read()
bbox = cv2.selectROI(frame)
ok = tracker.init(frame,bbox)

wall_bbox = cv2.selectROI(frame)
(x_wall,y_wall,x2_wall,y2_wall) = wall_bbox
print(wall_bbox)
num_cont_frames=0

#video = cv.VideoCapture(path)
ball_size = 0.22 #diameter of a regulation ball in meters
fps_cam = 10000 # Change this to the required fps of the video 
fps_vid =video.get(cv2.CAP_PROP_FPS)
fps_time= fps_vid / fps_cam 
#print(fps_time)


# Initialize variable to track state of ball (in contact with wall or not)
in_contact = False

# Initialize variable to track whether in_contact has ever been True
in_contact_ever = False


# Initialize lists for inbound and outbound velocities
inbound_velocities = []
outbound_velocities = []

# Calculate time interval between frames in seconds
time_interval = 1 / fps_cam

scale  = []
x_list = []
y_list = []
x_def=[]
inbound_x = []
inbound_y = []
outbound_x = []
outbound_y = []

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
       
        if x < max(x2_wall, x_wall): #sometimes the bbox is the wrong way around
            # Set in_contact to True
            in_contact = True
            # Set in_contact_ever to True
            in_contact_ever = True
            # Increment counter
            num_cont_frames = num_cont_frames + 1
            x_defe = x2-x2_wall
            x_def.append(x_defe) 
        else: 
           in_contact = False 

        if in_contact == False and in_contact_ever==False:
            inbound_x.append(x2) #list of x positions of right edge
            inbound_y.append(y2)  
        
        if in_contact == False and in_contact_ever==True:
            outbound_x.append(x2) #list of x positions of right edge
            outbound_y.append(y2)
    
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
x_speed=[]
y_speed=[]
for i in range(x2_len):
    x_speed=x_diff[i]*scale_ave*fps_cam
    y_speed=y_diff[i]*scale_ave*fps_cam


for i in range(distlen):
    speedcalc=realdist[i]*fps_cam
    speed.append(speedcalc)

contact_time=num_cont_frames/fps_cam
print(contact_time)

if x_def:
    realxdef = min(x_def)*scale_ave
else:
    realxdef = 0

print(realxdef)

# Create a dictionary with the data for the table
speeddata={'x_speed': x_speed, 'y_speed': y_speed, 'speed': speed, 'contact_time': contact_time, 'deformation' :realxdef}
# Create a pandas DataFrame with the data
df = pd.DataFrame(speeddata)
# Export the DataFrame to a CSV file
filename = file + 'results.csv'
df.to_csv(filename, index=False)
