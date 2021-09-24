import cv2 as cv
import numpy as np
import os


def preprocess_dataset(DIR, emotions=['Normal', 'Sad', 'Happy', 'Surprised'], haar_cascade_model='emotion_recognition/haar_face.xml'):
    """
        Scan the directory to and extract all faces from images and save as new a image
    """
    haar_cascade = cv.CascadeClassifier(haar_cascade_model)
    features, labels = [], []
    for label in emotions:
        path = os.path.join(DIR, label)
        label = emotions.index(label)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            print("+++++>", img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2BGRA)
            face_rects = haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=1)  #
            for i, (x, y, w, h) in enumerate(face_rects):
                faces_roi = gray[y:y+h, x:x+w]
                labels.append(label)
                features.append(faces_roi)
                cv.imwrite(os.path.join(img_path), img_array[y:y+h, x:x+w])
    print('=========DONE========')
    return features, labels


preprocess_dataset(DIR='emotion_recognition/Resources/Faces/New',
                   haar_cascade_model='emotion_recognition/haar_face.xml')
