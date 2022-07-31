import cv2
import numpy as np
from typing import List

class Distance():

    def __init__(self, real_distance, real_width):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        ref_image = cv2.imread('ref_image.jpg')

        gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        face = self.detector.detectMultiScale(gray, 1.3, 5)
        (_, _, _, self.ref_width) = face.ravel()

        self.real_distance = real_distance
        self.real_width = real_width

    def _focal_length(self, measured_dist: float, real_width: float, ref_width: int) -> float:
        focal_length = (ref_width / real_width) * measured_dist
        return focal_length

    def _calc_distance(self, focal_length: float, real_width: float, actual_width: int) -> float:
        distance = (real_width / actual_width) * focal_length
        return distance

    def get_distances(self, bounding_boxes: np.ndarray) -> List[int]:
        Focal_length = self._focal_length(self.real_distance, self.real_width, self.ref_width)

        distances = list()
        for bbox in bounding_boxes:
            (_, _, _, obj_width) = bbox

            if obj_width != 0:
                distances.append(self._calc_distance(Focal_length, self.real_width, obj_width))
            else:
                distances.append(0)

        return distances

DEVICE = 0

if __name__ == '__main__':
    dist = Distance(real_distance = 70, real_width = 15)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(DEVICE)
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 100)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    while(True):
        (ret, frame) = cap.read()

        #frame = cv2.rotate(frame, cv2.ROTATE_180)

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bounding_boxes = detector.detectMultiScale(gray, 1.3, 5)
            distances = dist.get_distances(bounding_boxes=bounding_boxes)

            for idx, (x, y, h, w) in enumerate(bounding_boxes):
                if idx == len(distances):
                    break
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame,"Distance: {:.2f}".format(distances[idx]), (x + 5, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 1255), 2)
            
            cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
        

    