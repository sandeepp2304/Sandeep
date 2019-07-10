# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 19:05:17 2018

@author: chandrashekar.k
"""

from imutils.video import FPS
import numpy as np
import imutils
import cv2
import datetime
import time

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

#stream = cv2.VideoCapture(0)
stream = cv2.VideoCapture('D://Opencv-POC//livevideo//2018-10-17_10-40-23_16.mp4')
#stream = cv2.VideoCapture('rtsp://admin:admin@192.168.50.28:80/cam/realmonitor?channel=1&subtype=0');
fps = FPS().start()

count=0;
while True:
    
    (grabbed, frame) = stream.read()
    
    if not grabbed:
        break
    
    frame = imutils.resize(frame,600)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])
    faces=faceDetect.detectMultiScale(frame,1.3,5)
    framedate=datetime.datetime.now() + datetime.timedelta(milliseconds=40)
    for(x,y,w,h) in faces:
        count = count+1;
        cv2.imwrite("dataSet/"+str(framedate).replace(":", "-").replace(".", "_")+".jpg",frame)
        #cv2.imwrite("dataSet/"+ time.strftime("%Y-%m-%d %H-%M-%S")+"_"+str(framedate)+ ".jpg",frame)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(10);
        
    cv2.putText(frame, "Fast Method", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    fps.update()
    
    
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


stream.release()
cv2.destroyAllWindows()    