import cv2 as cv
import numpy as np
import time



def nothing(x):
    pass


img = np.zeros((300,512,3), np.uint8)
cv.namedWindow('image')

# create trackbars for color change
cv.createTrackbar('SL','image',0,255,nothing)
cv.createTrackbar('HL','image',0,255,nothing)
cv.createTrackbar('VL','image',0,255,nothing)


# create trackbars for color change
cv.createTrackbar('SH','image',0,255,nothing)
cv.createTrackbar('HH','image',0,255,nothing)
cv.createTrackbar('VH','image',0,255,nothing)



cap = cv.VideoCapture("1552989328.51_rgb.avi");
if cap.isOpened()== False:
    print("error open video")
while(cap.isOpened()):
    ret, frame = cap.read()
    # frame = cv.flip(frame, 1)
    if ret == True:
        #cv.imshow('frame', frame)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hsv = hsv[:50,100:-100]
        cv.imshow('hsv', hsv)
        sl = cv.getTrackbarPos('SL','image')
        hl = cv.getTrackbarPos('HL','image')
        vl = cv.getTrackbarPos('VL','image')
        sh = cv.getTrackbarPos('SH','image')
        hh = cv.getTrackbarPos('HH','image')
        vh = cv.getTrackbarPos('VH','image')
        #lower_blue= np.array([95,50,150])
        #upper_blue = np.array([110,150,255])
        lower_blue= np.array([95,50,100])
        upper_blue = np.array([125,150,255])
        mask = cv.inRange(hsv, lower_blue, upper_blue)
        kernel = np.ones((3,3),np.uint8)
        mask = cv.erode(mask,kernel,iterations = 1)
        kernel = np.ones((11,11),np.uint8)
        mask =  cv.dilate(mask,kernel,iterations = 1)
        cv.imshow('mask',mask)

        img,cnts,hie = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        for cnt in cnts:
            x,y,w,h = cv.boundingRect(cnt)
            crop = frame[y:y+h,x+100:x+w+100]
            gray_crop = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
            gray_crop = cv.resize(gray_crop, (64, 64)) 
            cv.imwrite('right/right'+str(time.time())+'.jpg',gray_crop)
            cv.imshow('crop', crop)
        if cv.waitKey() & 0xFF == ord('q'):
            break
    else:
        break

    
cap.release()
cv.destroyAllWindows()