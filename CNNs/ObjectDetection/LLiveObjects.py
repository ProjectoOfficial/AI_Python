import cv2
import numpy as np
from numba import jit,cuda
def obj():
        net = cv2.dnn.readNet(r"yolov3.weights",r"yolov3.cfg")

        with open("coco.names","r") as f:
            classes = [line.strip() for line in f.readlines()]

        #print(classes)

        layers = net.getLayerNames()
        outLayers = [layers[i[0] -1] for i in net.getUnconnectedOutLayers()]

        cap = cv2.VideoCapture(0)

        while True:
            _, img = cap.read()
            height, width, channels = img.shape

            blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True,crop=False)

            #for b in blob:
            #    for n,image in enumerate(b):
            #        cv2.imshow("immagine ({})".format(n),image)

            net.setInput(blob)
            outs = net.forward(outLayers)

            #for o in outs:
            #    print(o)

            boxes = list()
            confidences = list()
            class_ids = list()

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)

                        x = int(center_x - w/2)
                        y = int(center_y - h/2)

                        boxes.append([x,y,h,w])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
            colors = np.random.uniform(0,255,size=(len(classes),3))
            for i in range(len(boxes)):
                if i in indexes:
                    x , y, h, w = boxes[i]
                    label = str(classes[class_ids[i]])
                    cv2.rectangle(img,(x,y),(x+w,y+h),colors[i],2)
                    cv2.putText(img,label,(x,y), cv2.FONT_HERSHEY_PLAIN,1,colors[i],1)

            cv2.imshow("foto",img)
            cv2.waitKey(10)

obj()