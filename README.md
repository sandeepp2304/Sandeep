# Sandeep
import cv2
import os
import time
import datetime
import shutil
from gevent.threadpool import ThreadPool
from time import sleep
import gevent
CONCURRENT = 1
import read_face_recogition_properties
import logging


property_keys=read_face_recogition_properties.read_face_recog_properties()
faceDetect=cv2.CascadeClassifier(property_keys['HAARCASCADE_FRONTALFACE_DEFAULT']);
log_file_path=property_keys['VIDEO_GRABBER_LOG_PATH']
logging.basicConfig(filename=log_file_path,level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')


def processVideoFiles(videoPath,name):
    gevent.sleep(2)
    count=0
    frameCount=0

    destPath=property_keys['DETECTED_IMGS_DEST_PATH']
    try:
        frame_date_time=name.split("_", 1)
        frame_date=frame_date_time[0]
        frameTime=frame_date_time[1].split("_", 1)[0].replace("-", ":")
        framefulldate = datetime.datetime.strptime(frame_date + ' ' + frameTime+".100000", "%Y-%m-%d %H:%M:%S.%f")
        
        video_capture = cv2.VideoCapture(videoPath)
        logging.info("Frame grabbing started for "+videoPath+" :name "+name)
        
        while True:
            framefulldate = framefulldate + datetime.timedelta(milliseconds=40)
            frameCount=frameCount+1
            logging.info("frameCount:"+str(frameCount))
            ret,frame = video_capture.read()
            if ret is False:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces=faceDetect.detectMultiScale(gray,1.3,5)
            if len(faces) > 0:
                count = count+1
                logging.info("Face detetced frame Count:"+str(count))
                cv2.imwrite(destPath+"/"+str(framefulldate).replace(":", "-").replace(".", "_")+".jpg", frame)
                
        video_capture.release()
        shutil.copy(videoPath,property_keys['PROCESSED_VIDEOS_DEST_PATH'])
        os.remove(videoPath)
        logging.info(videoPath+" was deleted successfully..")
        
    except Exception as e:
         logging.error("Error:"+str(e))
 

def processvideo():
    npool = ThreadPool(CONCURRENT)
    logging.info("thread pool started .....")
    path=property_keys['LIVE_VIDEOS_DEST_PATH']
    for path, subdirs, files in os.walk(path):
        del files[-1]

        for name in files:
            videoPath=os.path.join(path, name)
            #processVideoFiles(videoPath,name)
            npool.spawn(processVideoFiles,videoPath,name) 
    npool.join()      
    


def executeProcessVideoService():
    time.sleep(5)
    path=property_keys['LIVE_VIDEOS_DEST_PATH']
	
    list = os.listdir(path) # dir is your directory path
    try:
        if len(list) !=0 :
            processvideo()
            logging.info("video files found...")
        else:
            logging.info("No Video files found...")	
    
    except Exception as e:
        logging.error("Error:"+str(e))
        
			
while True:
    executeProcessVideoService()
	
if __name__ == '__main__':
	#executeProcessVideoService()
    processvideo()
