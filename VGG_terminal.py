# Made due to memory issues in JupyterLabs
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import random

from sklearn.datasets import load_files   
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras.saving import load_model
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler


with open('./Data/X_train.npy', 'rb') as f:
    X_train = np.load(f)
with open('./Data/y_train.npy', 'rb') as f:
    y_train = np.load(f)
with open('./Data/X_test.npy', 'rb') as f:
    X_test = np.load(f)
with open('./Data/y_test.npy', 'rb') as f:
    y_test = np.load(f)
    
dimensions = (200, 200)
num_classes = 5

# Necessary for ResNet Dense Layer
y_train_eye = np.eye(num_classes)[y_train.reshape(-1)]
y_test_eye = np.eye(num_classes)[y_test.reshape(-1)]


model = load_model('./ModelSaves/VGG16v1')


preds = model.evaluate(X_test, y_test_eye)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

preds = []
print("X_test size:", X_test.shape)
for i in range((X_test.shape[0] / 32) + 1):
    if i * 32 < X_test.shape[0]:
        preds.append(model(X_test[i * 32 : min((i + 1) * 32, X_test.shape[0])]))
preds = np.concatenate(preds)
print("preds size:", preds.shape)

emotions = ["Happiness", "Sadness", "Fear", "Disgust", "Anger", "Surprise"]

emotion_presence = set()

rands = []

shuffle_indices = shuffle(range(len(preds)))

for i in shuffle_indices:
    if emotions[classes[i]] not in emotion_presence and np.argmax(preds[i]) == classes[i]:
        rands.append(i)
        emotion_presence.add(emotions[classes[i]])

print(len(rands))

while len(rands) < 5:
    pos = random.randint(0, len(test_indices) - 1)
    if emotions[classes[test_indices[pos]]] not in emotion_presence:
        rands.append(pos)
        emotion_presence.add(emotions[classes[test_indices[pos]]])

# actuals = [test_indices[pos] for pos in rands]

for i in range(len(actuals)):
    print('========================================================')
    print(book_links[actuals[i]])
    # display(Image.fromarray((X_test[rands[i]] * 255).astype(np.uint8)))
    print(directory + str(actuals[i]) + ".jpg")
    print(emotions[np.argmax(preds[i])])
    # im = Image.open(directory + str(actuals[i]) + ".jpg")
    # im.thumbnail(dimensions, Image.ANTIALIAS)
    # display(im)
    print(emotions[classes[actuals[i]]])
    print('========================================================')
    
scores = {}
for i in range(len(y_test)):
    real_class = y_test[i]
    pred_class = np.argmax(preds[i])
    if real_class in scores:
        scores[real_class] = [scores[real_class][0] + int(real_class == pred_class), scores[real_class][1] + 1]
    else:
        scores[real_class] = [int(real_class == pred_class), 1]
        
for key in list(scores.keys()):
    print(key, scores[key], scores[key][0] / scores[key][1])