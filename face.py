import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
%matplotlib inline
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images
from deepface import DeepFace


videoFile = "vid1.mp4"
cap = cv2.VideoCapture(videoFile)

count = 0
frameRate = cap.get(5) #frame rate
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    
    result = DeepFace.analyze(frame)
    
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	faces = faceCascade.detectMultiScale(gray,1.1,4)

	for(x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0),2)
    

	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame, predictionss['dominant_emotion'], (0,50),font,1,(0,0,255), 2,cv2.LINE_4);
	plt.imshow(cv2.cvtColor(frame, cv2.cv2.COLOR_BGR2RGB))

    
cap.release()
print ("Done!")

#img = plt.imread('frame0.jpg')   # reading image using its name
#plt.imshow(img)

#data = pd.read_csv('mapping.csv')     # reading the csv file
#data.head()   


