import cv2
import os
import numpy as np

def faceDetection(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    haar_face=cv2.CascadeClassifier(r"C:\Users\daniel\PycharmProjects\tensorEnv\ELABORAZIONE_IMMAGINI\CV\haar\haarcascade_frontalface_default.xml")
    face= haar_face.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=6)

    return face,gray

def training_data(directory):
    faces=[]
    facesID=[]
    i: int
    i=0
    for path,subdir,filename in os.walk(directory):
        for filename in filename:
            print("TRAINING IMAGE ", i)
            i+=1
            if filename.startswith("."):
                print("skipping system file")
                continue
            id=os.path.basename(path)
            image_path=os.path.join(path,filename)
            img_test=cv2.imread(image_path)
            if img_test is None:
                print("error opening image")
                continue
            face,gray=faceDetection(img_test)
            if len(face) != 1:
                print(len(face))
                continue
            (x,y,w,h)=face[0]
            region=gray[y:y+w,x:x+h]
            faces.append(region)
            facesID.append(int(id))
    return faces,facesID


def train_classifier(faces,facesID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(facesID))
    return face_recognizer

def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)

def put_name(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.QT_FONT_NORMAL,2,(0,0,255),3)