import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import utils as np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import pickle as pk
import numpy as np

with open('./db/datas/cifar100/data_train.pkl', mode='rb') as f:
    data_train = pk.load(f)

with open('./db/datas/cifar100/data_test.pkl', mode='rb') as f:
    data_test = pk.load(f)

x_train = np.array([img['image'] for img in data_train])
y_train = np.array([label['label'] for label in data_train])
names = [name['label_name'] for name in data_train]

y_train = np_utils.to_categorical(y_train, num_classes=len(np.unique(y_train)))

generator_train = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, horizontal_flip=False)
generator_test = ImageDataGenerator()

base_train = generator_train.flow(x_train, y_train, batch_size=64)

model = Sequential()

model.add(InputLayer(shape=(32, 32, 3)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=100, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()


