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
n_components = 100
pca = PCA().fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()