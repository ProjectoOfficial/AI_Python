import cv2
from ELABORAZIONE_IMMAGINI.CV import facerecognition as fr

test_img = cv2.imread(r"C:\Users\daniel\PycharmProjects\tensorEnv\CV\immagini\0\img0.jpg")
face,gray=fr.faceDetection(test_img)

#faces,facesID=fr.training_data('C:/Users/daniel/PycharmProjects/tensorEnv/CV/immagini')
#faceRecognizer=fr.train_classifier(faces,facesID)
#faceRecognizer.save("trainedData.yml")

faceRecognizer=cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read(r"C:\Users\daniel\PycharmProjects\tensorEnv\ELABORAZIONE_IMMAGINI\CV\trainedData.yml")

name={0:"Elon", 1:"Daniel"}

for face in face:
    (x,y,w,h)=face
    region=gray[y:y+h,x:x+w]
    label, confidence= faceRecognizer.predict(region)
    print("confidence",confidence)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    print("Name: ", predicted_name)
    if(label != 0 and label != 1):
        print("non lo conosco")
    if(confidence>40):
        continue
    fr.put_name(test_img,predicted_name,x,y)
    cv2.imshow("face", test_img)
    if cv2.waitKey(10) == 27:
        break