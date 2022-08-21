import cv2
import random
import numpy as np

DEVICE = 0

if __name__ == "__main__":
    cap = cv2.VideoCapture(DEVICE)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5) 

    if not cap.isOpened():
        print("Cannot open video input device")
        exit(-1)

    colors = np.array([random.uniform(0, 256), random.uniform(0, 256), random.uniform(0, 256)], dtype=np.uint32).reshape(-1, 3)
    for i in range(99):
        r = random.uniform(0, 256)
        g = random.uniform(0, 256)
        b = random.uniform(0, 256)
        colors = np.concatenate((colors, np.array([random.uniform(0, 256), random.uniform(0, 256), random.uniform(0, 256)], dtype=np.uint32).reshape(-1, 3)))

    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
    
    _, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    mask = np.zeros_like(old_frame)
    
    lkparamters = dict( winSize = (30, 30),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        _, frame = cap.read()

        if frame.size == 0:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lkparamters)

        good_new = p1[status == 1]
        good_old = p0[status == 1]

        if good_new.size == 0 or good_old.size == 0:
            continue

        diff = good_new.mean(axis=1) - good_old.mean(axis=1)
        index = np.where(max(np.abs(diff))) if diff.any() != 0 else 0

        a, b = good_new[index, :].reshape(-1,2).astype(np.uint32).ravel()
        c, d = good_old[index, :].reshape(-1,2).astype(np.uint32).ravel()

        if a < frame.shape[1] and c < frame.shape[1] and b < frame.shape[0] and d < frame.shape[0]:
            mask = cv2.line(mask, (a, b), (c, d), colors[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, colors[i].tolist(), -1)

            if np.abs(diff[index]) > 0.5:
                if diff[index] < 0:
                    print("left")
                else:
                    print('right')

        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)

        if cv2.waitKey(1) == ord('q'):
            break

        if cv2.waitKey(1) == ord('r'):
            mask = np.zeros_like(old_frame)

        old_gray = gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()