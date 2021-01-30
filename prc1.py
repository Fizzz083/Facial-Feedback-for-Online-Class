import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt


img = cv2.imread('hb1.jpg');
predictionss = DeepFace.analyze(img)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

faces = faceCascade.detectMultiScale(gray,1.1,4)

for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y), (x+w, y+h), (0,255,0),2)
    

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, predictionss['dominant_emotion'], (0,50),font,1,(0,0,255), 2,cv2.LINE_4);
plt.imshow(cv2.cvtColor(img, cv2.cv2.COLOR_BGR2RGB))



