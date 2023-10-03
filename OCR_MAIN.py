from keras.datasets import mnist
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import matplotlib.pyplot as plt
import argparse
import cv2
import os
from collections import OrderedDict
from sklearn.preprocessing import LabelBinarizer

folders = []

path = 'Data/EnglishImg/English/Img/GoodImg/Bmp'

for root, dirnames, filenames in os.walk(path):
  for j in dirnames:
    folders.append(j)

files = {}

dict1 = OrderedDict(sorted(files.items()))

data = []
labels = []
tmp = 0

img = cv2.imread('Data/EnglishImg/English/Img/GoodImg/Bmp/Sample041/img041-00190.png')


def conTO28x28(path):
  img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

  # convert each image of shape (32, 28, 1)
  w, h = img.shape
  if h > 28 or w > 28:
    (tH, tW) = img.shape
    dX = int(max(0, 28 - tW) / 2.0)
    dY = int(max(0, 28 - tH) / 2.0)

    img = cv2.copyMakeBorder(img, top=dY, bottom=dY,
                             left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                             value=(0, 0, 0))
    img = cv2.resize(img, (28, 28))

  w, h = img.shape

  if w < 28:
    add_zeros = np.ones((28 - w, h)) * 255
    img = np.concatenate((img, add_zeros))

  if h < 28:
    add_zeros = np.ones((28, 28 - h)) * 255
    img = np.concatenate((img, add_zeros), axis=1)
  return img


for i in dict1.keys():
  for j in dict1[i]:
    labels.append(tmp)
    image = conTO28x28("Data/EnglishImg/English/Img/GoodImg/Bmp/" + i + "/" + j)
    data.append(image)
  tmp += 1

from numpy import save

labels = np.array(labels, dtype="int")
data = np.array(data, dtype='float32')

save('data.npy', data)
save('labels.npy', labels)

tmp_data = np.load('data.npy')


def load_az_dataset(datasetPath):
  data = []
  labels = []
  for row in open(datasetPath):
    row = row.split(",")
    label = int(row[0])
    image = np.array([int(x) for x in row[1:]], dtype="uint8")
    image = image.reshape((28, 28))
    data.append(image)
    labels.append(label)
  data = np.array(data, dtype="float32")
  labels = np.array(labels, dtype="int")
  return (data, labels)


((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
(azData, azLabels) = load_az_dataset("Data/A_Z Handwritten Data.csv")

mnist_data = np.vstack([trainData, testData])
mnist_labels = np.hstack([trainLabels, testLabels])

labels1 = np.hstack([mnist_labels, labels, azLabels])
data1 = np.vstack([mnist_data, data, azData])

save("Data/combined_data.npy", data1)
save("Data/combined_labels.npy", labels1)

loaded_data = np.load('Data/combined_data.npy')
loaded_labels = np.load('Data/combined_labels.npy')

data = [cv2.resize(image, (32, 32)) for image in loaded_data]
data = np.array(data, dtype='float32')

data2 = np.expand_dims(data, axis=-1)

le = LabelBinarizer()
labels = le.fit_transform(loaded_labels)
counts = labels.sum(axis=0)

classTotals = labels.sum(axis=0)
classWeight = {}

for i in range(0, len(classTotals)):
  classWeight[i] = classTotals.max() / classTotals[i]

(trainX, testX, trainY, testY) = train_test_split(data2,
                                                  labels, test_size=0.25, stratify=labels, random_state=42)

aug = ImageDataGenerator(
  rotation_range=10,
  zoom_range=0.05,
  width_shift_range=0.1,
  shear_range=0.15,
  height_shift_range=0.1,
  fill_mode="nearest"
)

import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Conv2D, AveragePooling2D, Flatten, Dense
from keras.layers import add
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

def residual_module(data, K, stride, chanDim, red=False,
                    reg=0.0001, bnEps=2e-5, bnMom=0.9):
    shortcut = data

    bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                             momentum=bnMom)(data)
    act1 = Activation("relu")(bn1)
    conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
                   kernel_regularizer=l2(reg))(act1)

    bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                             momentum=bnMom)(conv1)
    act2 = Activation("relu")(bn2)
    conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,
                   padding="same", use_bias=False,
                   kernel_regularizer=l2(reg))(act2)

    bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                             momentum=bnMom)(conv2)
    act3 = Activation("relu")(bn3)
    conv3 = Conv2D(K, (1, 1), use_bias=False,
                   kernel_regularizer=l2(reg))(act3)

    if red:
        shortcut = Conv2D(K, (1, 1), strides=stride,
                           use_bias=False, kernel_regularizer=l2(reg))(act1)

    x = add([conv3, shortcut])
    return x

def build_resnet(inputShape, classes, stages, filters,
                 reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar"):
    chanDim = -1
    if tf.keras.backend.image_data_format() == "channels_first":
        inputShape = (inputShape[2], inputShape[0], inputShape[1])
        chanDim = 1

    inputs = Input(shape=inputShape)
    x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                           momentum=bnMom)(inputs)

    if dataset == "cifar":
        x = Conv2D(filters[0], (3, 3), use_bias=False,
                   padding="same", kernel_regularizer=l2(reg))(x)
    elif dataset == "tiny_imagenet":
        x = Conv2D(filters[0], (5, 5), use_bias=False,
                   padding="same", kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                               momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((3, 3), strides=(2, 2))(x)

    for i in range(0, len(stages)):
        stride = (1, 1) if i == 0 else (2, 2)
        x = residual_module(x, filters[i + 1], stride,
                            chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

        for j in range(0, stages[i] - 1):
            x = residual_module(x, filters[i + 1],
                                (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

    x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                           momentum=bnMom)(x)
    x = Activation("relu")(x)
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    x = Dense(classes, kernel_regularizer=l2(reg))(x)
    x = Activation("softmax")(x)

    model = Model(inputs, x, name="resnet")
    return model

def train_resnet_model(model, trainX, trainY, testX, testY, classWeight, EPOCHS, INIT_LR, BS, checkpoint_path):
    print("[INFO] compiling model...")
    opt = SGD(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    aug = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.05,
        width_shift_range=0.1,
        shear_range=0.15,
        height_shift_range=0.1,
        fill_mode="nearest"
    )

    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS,
        class_weight=classWeight,
        verbose=1,
        callbacks=[cp_callback]
    )

    model.save("models/RESNET_OCR", save_format='.h5')
    print("Model Saved in the path: mode;s/RESNET_OCR")



# Define the ResNet model
inputShape = (32, 32, 1)  # Modify this according to your data
classes =  len(le.classes_) # Number of classes 
stages = [3, 3, 3]  # Modify as needed
filters = [64, 64, 128, 256]  # Modify as needed
model = build_resnet(inputShape, classes, stages, filters)

checkpoint_path = "../models"


# Train the ResNet model
train_resnet_model(model, trainX, trainY, testX, testY, classWeight, EPOCHS=25, INIT_LR = 1e-1, BS=256, checkpoint_path = "../models")

