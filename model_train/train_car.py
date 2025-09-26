import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Rescaling, GlobalAveragePooling2D
from tensorflow.keras import utils as np_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import pickle as pk
import numpy as np
from google.colab import drive

drive.mount('/content/drive')

with open('/content/drive/MyDrive/datas/cars/data_train.pkl', mode='rb') as f:
    data_train, names = pk.load(f)

with open('/content/drive/MyDrive/datas/cars/data_test.pkl', mode='rb') as f:
    data_test, names = pk.load(f)

x_train = np.array([img['img'] for img in data_train])
y_train = np.array([label['label'] for label in data_train])
print(len(np.unique(y_train)))
y_train = np_utils.to_categorical(y_train, num_classes=len(np.unique(y_train)))
generator_train = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, horizontal_flip=False)
base_train = generator_train.flow(x_train, y_train, batch_size=128)

print(x_train.shape)

x_test = np.array([img['img'] for img in data_test])
y_test = np.array([label['label'] for label in data_test])
y_test = np_utils.to_categorical(y_test, num_classes=len(np.unique(y_test)))

generator_test = ImageDataGenerator()
base_test = generator_test.flow(x_test, y_test, batch_size=128)

model = Sequential()

model.add(Rescaling(1./255, input_shape=(256, 256, 3)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(GlobalAveragePooling2D())

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))


model.add(Dense(units=48, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

callback_list = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        min_delta=0.00005 ,

    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        patience=5,
        factor=0.90,
        min_lr=1e-5
    )
]

history = model.fit(base_train, epochs=250, shuffle=True, callbacks=callback_list, validation_data=base_test)
model.save('/content/drive/MyDrive/models/cars/modelV6.keras')

model.evaluate(base_test)

print(history.history.keys())

plt.figure(figsize=(8, 8))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('loss during training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('accuracy during training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()