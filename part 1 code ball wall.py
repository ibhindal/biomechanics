import cv2
import numpy as np
import os


os.system('cls')
os.chdir("C:\\Users\\Ibrahim\\Desktop\\biomechancs module")


tracker = cv2.TrackerKCF_create()
video = cv2.VideoCapture('video.mp4')
ok,frame=video.read()
bbox = cv2.selectROI(frame)
ok = tracker.init(frame,bbox)

fps_cam = 1500 # Change this to the required fps of the video 
fps_vid =video.get(cv2.CAP_PROP_FPS)
fps_time= fps_vid / fps_cam
print(fps_time)

file = open('videocoord.txt', 'w')

ball_size = 22 #diameter of a regulation ball

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
        scale1= ball_d/ball_size  
        scale=[]
        scale.append(scale1)



        for listitem in text1:
            file.write(f'{listitem}\n')       
   else:
        cv2.putText(frame,'Error',(100,0),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
   cv2.plt.imshow('Tracking',frame)
   if cv2.waitKey(1) & 0XFF==27:
        break


print (scale)
file.close()

cv2.destroyAllWindows()