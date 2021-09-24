from posixpath import join
import cv2 as cv
import torch
from torchvision import datasets, transforms
import numpy as np
import time
import os
from mtcnn.mtcnn import MTCNN

face_model = MTCNN()


def preprocess_dataset(DIR='emotion_recognition/Resources/Faces/Patrik', emotions=['Happy', 'Normal', 'Sad', 'Surprised']):
    for label in emotions:
        path = os.path.join(DIR, label)
        label = emotions.index(label)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            print("+++++>", img_path)
            faces = face_model.detect_faces(img_array)
            for face in faces:
                x, y, w, h = face['box']
                faces_roi = img_array[y:y+h, x:x+w]
                cv.imwrite(os.path.join(img_path), faces_roi)
    print('=========DONE========')


preprocess_dataset(DIR='emotion_recognition/Resources/Faces/Labels')
