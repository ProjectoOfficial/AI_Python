import cv2
from ELABORAZIONE_IMMAGINI.CV import facerecognition as fr
import numpy as np
#faces,facesID=fr.training_data(r'C:\Users\daniel\PycharmProjects\tensorEnv\ELABORAZIONE_IMMAGINI\CV\immagini')
#faceRecognizer=fr.train_classifier(faces,facesID)
faceRecognizer=cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read(r"C:\Users\daniel\PycharmProjects\tensorEnv\ELABORAZIONE_IMMAGINI\CV\trainedData.yml")

name = {0: "Elon", 1: "Daniel"}

capture = cv2.VideoCapture(0)
i: int
i=0
while True:
    ret, test_img = capture.read()
    faces_detected, gray = fr.faceDetection(test_img)

    for face in faces_detected:
        (x, y, w, h) = face
        region = gray[y:y+w, x:x+h]
        label, confidence = faceRecognizer.predict(region)
        print("confidence", confidence)
        predictedName = name[label]
        fr.draw_rect(test_img, face)
        if confidence > 35 and confidence!=0:
            continue
        fr.put_name(test_img, predictedName, x, y)
        f = []
        region = gray[y:y + w, x:x + h]
        f.append(region)
        n = []
        n.append(int(label))
        faceRecognizer.train(f, np.array(n))

    resized = cv2.resize(test_img, (800, 800))

    cv2.imshow("face", resized)
    if cv2.waitKey(10) == 27:
        break
faceRecognizer.save("trainedData.yml")