import cv2

def click_event(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDOWN:
    #print(f'Original Coordinates : ({x},{y})')

    xy = x/frame.shape[1]
    yy = y/frame.shape[0]
    print(f'YOLO Coordinates : ({xy},{yy})')
    cv2.putText(frame, f'({x},{y})',(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.circle(frame, (x,y), 8, (255,0,0), -1)
    return xy , yy
    

videopath = "CL_1_S0003.mp4"

videcap = cv2.VideoCapture(videopath)

cv2.namedWindow("1")
cv2.setMouseCallback("1",click_event)    
            
while(videcap.isOpened()):
    ret, frame = videcap.read()
    if ret == True:
        cv2.imshow('1', frame)   

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
