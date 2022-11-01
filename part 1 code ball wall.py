import cv2
import numpy as np
import os


# Clears command window 
os.system('cls')
# Changes directory
os.chdir("C:\\Users\\Ibrahim\\Desktop\\biomechancs module")
# Loads the tracker
tracker = cv2.TrackerKCF_create()
# Loads a video
video = cv2.VideoCapture('P2_1_30_1.mp4')
#function to read video frame by frame
ok,frame=video.read()
# Sets up a variable called bounding box for user to selecct ROI
bbox = cv2.selectROI(frame)
# Creates a mask for ROI
ok = tracker.init(frame,bbox)

fps_cam = 1500 # Change this to the required fps of the video 
fps_vid =video.get(cv2.CAP_PROP_FPS)
fps_time= fps_vid / fps_cam
print(fps_time)

file = open('videocoord.txt', 'w')

ball_size = 22  

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
        text1=['aa' ,x ,y ,x2 ,y2]
        ball_d= x2-x
        print(ball_d)
        scale1= ball_d/ball_size  
        scale=[]
        scale.append(scale1)
          #for listitem in text1:
            #file.write(f'{listitem}\n')       
   else:
        cv2.putText(frame,'Error',(100,0),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
   cv2.imshow('Tracking',frame)
   if cv2.waitKey(1) & 0XFF==27:
        break


print (scale)
file.close()

cv2.destroyAllWindows()