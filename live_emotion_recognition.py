import cv2 as cv
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
#from mtcnn.mtcnn import MTCNN
#import mediapipe as mp


def preprocess_image(image):
    # swap the color channels from BGR to RGB, resize it, and scale
    # the pixel values to [0, 1] range
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (64, 64))
    image = image.astype("float32") / 255.0
    # subtract ImageNet mean, divide by ImageNet standard deviation,
    # set "channels first" ordering, and add a batch dimension
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    # return the preprocessed image
    return torch.from_numpy(image)


model = torch.load('Trained_Model')
model.eval()

labels = ['Happyness', 'Neutral', 'Sadness',
          'Surprise']  # Fear, Anger, Contempt
colors = [(0, 255, 0), (255, 240, 0), (0, 0, 255),  (255, 0, 255)]

haar_cascade = cv.CascadeClassifier('haar_face.xml')
capture = cv.VideoCapture(0)
pTime = time.time()

while True:
    isTrue, img = capture.read()
    face_rect = haar_cascade.detectMultiScale(
        img, minNeighbors=5)
    for (x, y, w, h) in face_rect:
        face_roi = img[y:y+h, x:x+w]
        resized_tensor = preprocess_image(face_roi)
        prob, label = model.predict_proba(resized_tensor)
        # print(label)
        cv.rectangle(img, (x, y), (x+w, y+h), colors[label], thickness=1)
        cv.putText(img, f'Emotion: {labels[label]}({prob.item():.2F})', (x, y-10),
                   cv.FONT_HERSHEY_DUPLEX, 1, colors[label])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, f'{fps:.2F}', (20, 20),
               cv.FONT_HERSHEY_PLAIN, .8, (0, 255, 0), 2)
    cv.imshow("VIDEO", img)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
