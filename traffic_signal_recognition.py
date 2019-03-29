import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import cv2
import numpy as np
import glob
import time
import pickle


def nothing(x):
    pass


img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('SL','image',0,255,nothing)
cv2.createTrackbar('HL','image',0,255,nothing)
cv2.createTrackbar('VL','image',0,255,nothing)


# create trackbars for color change
cv2.createTrackbar('SH','image',0,255,nothing)
cv2.createTrackbar('HH','image',0,255,nothing)
cv2.createTrackbar('VH','image',0,255,nothing)



left_list = [cv2.cvtColor( cv2.imread(file), cv2.COLOR_BGR2GRAY).flatten() for file in glob.glob('left/*.jpg')]
right_list = [cv2.cvtColor( cv2.imread(file), cv2.COLOR_BGR2GRAY).flatten() for file in glob.glob('right/*.jpg')]
unknown_list = [cv2.cvtColor( cv2.imread(file), cv2.COLOR_BGR2GRAY).flatten() for file in glob.glob('unknown/*.jpg')]


X =  left_list+ right_list+ unknown_list;
Y = np.zeros(len(X))
X = np.asarray(X)
print(X.shape)
Y[0:len(left_list)-1]= 1
Y[-(len(unknown_list)-1):] = 2
# Load data

h, w = 64,64
target_names = ["right", "left", "unknown"]
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
# Compute a PCA 
n_components = 150
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

# apply PCA transformation to training data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
# train a neural network
print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes=(1024), batch_size=256, verbose= True, early_stopping=True).fit(X_train_pca, y_train)
img = X_test_pca[1,:]

y_pred = clf.predict(X_test_pca)
#print(y_pred)   
print(classification_report(y_test, y_pred, target_names=target_names))

pickle.dump(pca, open("pca.sav", "wb"), protocol=2)
pickle.dump(clf, open("clf.sav", "wb"), protocol=2)
print("Saved")

cap = cv2.VideoCapture('1552986091.26_rgb.avi');
if cap.isOpened()== False:
    print("error open video")
while(cap.isOpened()):
    start_time = time.time()
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame,1)
        sl = cv2.getTrackbarPos('SL','image')
        hl = cv2.getTrackbarPos('HL','image')
        vl = cv2.getTrackbarPos('VL','image')
        
        sh = cv2.getTrackbarPos('SH','image')
        hh = cv2.getTrackbarPos('HH','image')
        vh = cv2.getTrackbarPos('VH','image')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = hsv[0:100,100:-100]
        lower_blue= np.array([90,50,150])
        upper_blue = np.array([110,150,255])
        #lower_blue= np.array([90 ,100 ,80])
        #upper_blue = np.array([140 ,255 ,140])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.erode(mask,kernel,iterations = 1)
        kernel = np.ones((13,13),np.uint8)
        mask =  cv2.dilate(mask,kernel,iterations = 1)
        cv2.imshow('bin', mask)
        img,cnts,hie = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            if (w< 20 or h< 30):
                continue
            if (w > 100 or h > 100):
                continue
            crop = frame[y:y+h,x+100:x+w+100]
            
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray_crop = cv2.resize(gray_crop, (64, 64))
            gray_crop -=30;
            cv2.imshow('crop',crop)  
            flatten = gray_crop.flatten()
            flatten_trans = pca.transform([flatten])
            pred = clf.predict(flatten_trans)
            pred = pred.astype(int)
            font = cv2.FONT_HERSHEY_SIMPLEX
            print(target_names[pred[0]])
            #cv2.rectangle(frame,(x+100,y),(w,h),(0,255,0),3)
            cv2.putText(frame, target_names[pred[0]], (x+100,y), font, 0.8, (0, 255, 0),2, cv2.LINE_AA)
        cv2.imshow('frame',frame)
        end_time = time.time() -start_time;
        if end_time != 0:
            print("fps: ", 1.0/end_time)
        if cv2.waitKey() & 0xFF == ord('q'):
            break
    else:
        break

    
cap.release()
cv2.destroyAllWindows()
