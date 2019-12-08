import cv2 as cv
import numpy as np
import time
import sys
import pickle

def nothing(x):
    pass

if __name__ == '__main__':
    args = sys.argv

    cv.namedWindow('hsv')
    # create trackbars for color change
    cv.createTrackbar('HL','hsv',0,179,nothing)
    cv.createTrackbar('SL','hsv',0,255,nothing)
    cv.createTrackbar('VL','hsv',0,255,nothing)
    # create trackbars for color change
    cv.createTrackbar('HH','hsv',0,179,nothing)
    cv.createTrackbar('SH','hsv',0,255,nothing)
    cv.createTrackbar('VH','hsv',0,255,nothing)

    cv.setTrackbarPos('HL', 'hsv', 100)
    cv.setTrackbarPos('HH', 'hsv', 120)
    cv.setTrackbarPos('SL', 'hsv', 200)
    cv.setTrackbarPos('SH', 'hsv', 255)
    cv.setTrackbarPos('VL', 'hsv', 0)
    cv.setTrackbarPos('VH', 'hsv', 255)

    image_path = args[1]
    frame = cv.imread(image_path)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    target_names = ["right", "left", "unknown"]
    pca = pickle.load(open("pca.sav","rb"))
    clf = pickle.load(open("clf.sav", "rb"))
    font = cv.FONT_HERSHEY_SIMPLEX
    hl = 100
    hh = 120
    vl = 0
    vh = 255

    while True:
    #for i in range(255, 45, -30):
        # H, S, V
        sl = cv.getTrackbarPos('SL','hsv')
        #hl = cv.getTrackbarPos('HL','hsv')
        #vl = cv.getTrackbarPos('VL','hsv')
        sh = cv.getTrackbarPos('SH','hsv')
        #hh = cv.getTrackbarPos('HH','hsv')
        #vh = cv.getTrackbarPos('VH','hsv')
    
        lower_blue = np.array([hl, sl, vl])
        upper_blue = np.array([hh, sh, vh])
        mask = cv.inRange(hsv, lower_blue, upper_blue)

        cnts, _ = cv.findContours(mask,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cnts_vector = []
        cnts_position = []
        

        for i, cnt in enumerate(cnts):
            x,y,w,h = cv.boundingRect(cnt)

            gray_crop = gray[y:y+h,x:x+w]
            gray_crop = cv.resize(gray_crop, (64, 64))
        
            flatten = gray_crop.flatten()
            flatten_trans = pca.transform([flatten])
            cnts_vector.append(flatten_trans[0])
            cnts_position.append((x, y, w, h))
    
        result = frame.copy()
        if len(cnts_vector) > 0:
            pred = clf.predict_proba(cnts_vector)
        
            for i in range(len(pred)):
                prob = pred[i]
                index = prob.argmax()
                if index == 2:
                    continue

                text = target_names[index]
                x, y, w, h = cnts_position[i]
                cv.putText(result, text, (x, y - 10), font, 1, (0, 0, 255), 2, cv.LINE_AA)
                cv.rectangle(result, (x, y), (x + w, y + h), (255, 0,0 ), 2)
       
        res = np.hstack((hsv, cv.merge([mask, mask, mask]), result))
        res = cv.resize(res, (1300, 500))
        cv.imshow('hsv', res)
        k = cv.waitKey(500)
        if k == ord('q'):
            print('Exit')
            break
    
cv.destroyAllWindows()
