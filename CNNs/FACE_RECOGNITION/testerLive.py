import cv2
import os
import numpy as np
from CV import faceRecognition as fr

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r"C:\Users\daniel\PycharmProjects\tensorEnv\CV\trainingData.yml")

name= {0:"Elon",1:"Daniel"}

cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()
    faces_detected,gray_img=fr.faceDetection(test_img)

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)

    cv2.imshow("face detection ", test_img)
    cv2.waitKey(10)

    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w,x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)
        print("confidence: ",confidence)
        print("label", label)

        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        if confidence < 40:
            fr.put_text(test_img,predicted_name,x,y)

    cv2.imshow('face recognition', test_img)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
