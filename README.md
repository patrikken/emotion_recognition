# Facial Emotion Recognition

This repository contains the source code for the tutorial on facial emotion recognition. The project is designed taking into account all the steps of a machine learning project. Data collection, data labeling and preprocessing, model training and parameter tuning and finally deployment of the model to a real world scenario.
The required packages are pytorch, OpenCV, torchvision, numpy and matplotlib.

![output example](./example/gif.gif)

### Install dependancies 
```
pip install pytorch opencv torchvision numpy matplotlib
```


### Run pretrained model for emotion recognition from your webcam video
To test the final render of the project with the pretrained model run the file live_emotion_recognition.py:
```
python3 live_emotion_recognition.py
```

### Jupyter Notebook
The notebook file **notebook.ipynb** contains all the steps for the model training

### Project structure
The project is structured as follow:
- We first generate our own dataset using our webcam. With opencv, we will capture different face pause ('Happy', 'Normal', 'Sad', 'Surprised') and store them in different folders named with corresponding labels.
- We preprocess the images by extracting faces using pretrained face detector model (haar_cascade). Other face detector algorithms such as MTCNN or Mediapipe can be used. The dataset is then splited into test and valitation sets in different folders.
- We then build a CNN model with Pytorch to classify face emotions ![model](./model.py). The model is then trained on the training set and saved. 
- Finally with used the trained model with opencv to recognize facial emotion on live streams videos. 
