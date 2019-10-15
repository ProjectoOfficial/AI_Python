import cv2
from ELABORAZIONE_IMMAGINI.CV import facerecognition as fr

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read(r"C:\Users\daniel\PycharmProjects\tensorEnv\ELABORAZIONE_IMMAGINI\CV\trainedData.yml")

name = {0:"Elon", 1:"Daniel"}

capture = cv2.VideoCapture(0)

while True:
    ret, test_img=capture.read()
    faces_detected,gray=fr.faceDetection(test_img)

    #for (x,y,w,h) in faces_detected:
    #    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,0,255),thickness=3)

    #resized = cv2.resize(test_img,(600,600))
    #cv2.imshow("face",resized)
    #cv2.waitKey(10)

    for face in faces_detected:
        (x,y,w,h)=face
        region=gray[y:y+w,x:x+h]
        label,confidence = faceRecognizer.predict(region)
        print("confidence",confidence)
        predictedName=name[label]
        fr.draw_rect(test_img,face)
        if confidence>35:
            continue
        fr.put_name(test_img,predictedName,x,y)
    resized = cv2.resize(test_img, (800, 800))

    cv2.imshow("face", resized)
    if cv2.waitKey(10) == 27:
        break