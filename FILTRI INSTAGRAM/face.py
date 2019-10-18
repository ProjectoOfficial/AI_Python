import cv2
import dlib

def Effect1():
    cap = cv2.VideoCapture(0)

    eye_png = cv2.imread(r'C:\Users\daniel\PycharmProjects\tensorEnv\ELABORAZIONE_IMMAGINI\INSTAGRAM_EFFECTS\images\eyes\blue_eye.png')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r'C:\Users\daniel\PycharmProjects\tensorEnv\ELABORAZIONE_IMMAGINI\INSTAGRAM_EFFECTS\shape_predictor_68_face_landmarks.dat')

    while True:
        ret, frame = cap.read()

        try:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        except:
            print("fotocamera non collegata")
            break
        faces = detector(frame)

        for face in faces:
            landmarks = predictor(gray,face)

            tlel = landmarks.part(37).x, landmarks.part(37).y
            brel = landmarks.part(40).x, landmarks.part(40).y

            tler = landmarks.part(43).x, landmarks.part(43).y
            brer = landmarks.part(46).x, landmarks.part(46).y

            #cv2.circle(frame,tlel,3,(255,0,0),-1)
            #cv2.circle(frame,brel,3,(255,0,0),-1)

            #occhio sinistro dimensioni
            l_eye_width = brel[0]-tlel[0]
            l_eye_heght = brel[1]-tlel[1]

            # occhio destro dimensioni
            r_eye_width = brer[0] - tler[0]
            r_eye_heght = brer[1] - tler[1]

            eye1 = cv2.resize(eye_png,(int(l_eye_width),int(l_eye_heght)))
            eye_area1 = frame[int(tlel[1]): int (brel[1]), int(tlel[0]): int(brel[0])]

            eye2 = cv2.resize(eye_png, (int(r_eye_width), int(r_eye_heght)))
            eye_area2 = frame[int(tler[1]): int(brer[1]), int(tler[0]): int(brer[0])]

            left_eye_gray = cv2.cvtColor(eye1,cv2.COLOR_BGR2GRAY)
            r1, left_eye_mask = cv2.threshold(left_eye_gray,25,255,cv2.THRESH_BINARY_INV)
            left_eye_area = cv2.bitwise_and(eye_area1,eye_area1, mask=left_eye_mask)

            right_eye_gray = cv2.cvtColor(eye2, cv2.COLOR_BGR2GRAY)
            r2, right_eye_mask = cv2.threshold(right_eye_gray, 25, 255, cv2.THRESH_BINARY_INV)
            right_eye_area = cv2.bitwise_and(eye_area2, eye_area2, mask=right_eye_mask)

            left_eye_final = cv2.add(left_eye_area,eye1)
            frame[int(tlel[1]): int(brel[1]), int(tlel[0]): int(brel[0])] = left_eye_final

            right_eye_final = cv2.add(right_eye_area, eye2)
            frame[int(tler[1]): int(brer[1]), int(tler[0]): int(brer[0])] = right_eye_final

        cv2.imshow("Frame",frame)

        if cv2.waitKey(10)==27:
            break

Effect1()