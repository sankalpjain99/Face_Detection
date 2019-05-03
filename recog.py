import cv2
import sys

faceCascade = cv2.CascadeClassifier('S:\Study material\Face_recognition\haarcascade_frontalface_default.xml')
#haarcascade_frontalface_default
videocap = cv2.VideoCapture(cv2.CAP_DSHOW)

while True:
    ret,frame = videocap.read()
    if ret!=0:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30,30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
        cv2.imshow('Video',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

videocap.release()
cv2.destroyAllWindows()
    