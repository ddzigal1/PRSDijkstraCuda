import cv2
import numpy as np
import math

# sd = ShapeDetector()
face_cascade = cv2.CascadeClassifier('face2.xml')
video = cv2.VideoCapture(0)
hueLower=3
hueUpper=33
mask = np.zeros((720, 1280), np.uint8)
bg = np.zeros((1, 65), np.float64)
fg = np.zeros((1, 65), np.float64)
fgbg = cv2.createBackgroundSubtractorMOG2()

while video.isOpened():
    ret, img2 = video.read()


    img = cv2.GaussianBlur(img2, (11, 11), 5)
    faces = face_cascade.detectMultiScale(img2, 1.3, 5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.inRange(img, (3, 50, 50), (15, 255, 255))
    img = cv2.medianBlur(img, 11)
    img = cv2.erode(img, None, iterations=1)
    if faces != ():
        for face in faces:
            (x, y, w, h) = face
            img[y - 50:y + h, x:x + w] = 0

    img3 = np.zeros(img.shape)


    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if contours!=[]:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        img3[y:y+h, x:x+w] = cv2.medianBlur(img[y:y+h, x:x+w], 11)
        cv2.rectangle(img3, (x, y), (x+w, y+h), 255)


        hull = cv2.convexHull(c, returnPoints=False)
        defects = cv2.convexityDefects(c, hull)
        cnt = 0
        if hasattr(defects,'shape'):
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(c[s][0])
                end = tuple(c[e][0])
                far = tuple(c[f][0])
                cv2.line(img3, start, end, 255, 2)
                cv2.circle(img3, far, 5, [0, 0, 255], -1)
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                ci = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + ci ** 2 - a ** 2) / (2 * b * ci))
                if angle <= math.pi / 2:
                    cnt += 1
                    cv2.circle(img3, far, 8, 255, -1)

        cnt = 0 if not cnt else cnt+1
        st = "Kamen"
        if cnt >=4:
            st = "Papir"
        elif cnt >=2:
            st = "Makaze"
        elif cnt >=1:
            st = "Kamen"


        cv2.putText(img3, st, (x,y-50), cv2.FONT_HERSHEY_SIMPLEX, 3,255)
        #print(cnt)

    cv2.drawContours(img2, contours, -1, (255, 255, 0), 3)
    cv2.imshow('bez_filtera',img2)
    cv2.imshow('KPM', img3)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
