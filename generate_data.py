import cv2 as cv
import os
import time
from mtcnn.mtcnn import MTCNN

DIR = 'emotion_recognition/Resources/Faces/Labels'


haar_cascade = cv.CascadeClassifier('emotion_recognition/haar_face.xml')

face_detector = MTCNN()

capture = cv.VideoCapture(0)
while True:
    key = cv.waitKey(20) & 0xFF
    isTrue, img = capture.read()

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # faces_rect = haar_cascade.detectMultiScale(gray)  # , scaleFactor=1.1, minNeighbors=1

    faces = face_detector.detect_faces(img)

    img_copy = img.copy()
    for face in faces:
        x, y, w, h = face['box']
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=1)

    """ 
    if(key == ord('s')):
        name = "{:.1f}".format(time.time())
        path = os.path.join(DIR, f'Smile/{name}.jpg')
        cv.imwrite(path, img_copy)
        print(f"IMAGE => {name}.jpg") """

    if(key == ord('t')):
        # press on t save as Sad image
        name = "{:.1f}".format(time.time())
        path = os.path.join(DIR, f'Sad/{name}.jpg')
        cv.imwrite(path, img_copy)
        print(f"IMAGE => {name}.jpg")

    if(key == ord('h')):
        # press on h save as Happy image
        name = "{:.1f}".format(time.time())
        path = os.path.join(DIR, f'Happy/{name}.jpg')
        cv.imwrite(path, img_copy)
        print(f"IMAGE => {name}.jpg")

    if(key == ord('n')):
        # press on n save as Normal image
        name = "{:.1f}".format(time.time())
        path = os.path.join(DIR, f'Normal/{name}.jpg')
        cv.imwrite(path, img_copy)
        print(f"IMAGE => {name}.jpg")

    if(key == ord('s')):
        # press on 'e' save as Surprised image
        name = "{:.1f}".format(time.time())
        path = os.path.join(DIR, f'Surprised/{name}.jpg')
        cv.imwrite(path, img_copy)

    cv.imshow("VIDEO", img)

    if key == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
