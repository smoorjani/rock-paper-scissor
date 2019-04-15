'''
PackHacks Rock Paper Scissors

A computer-vision based version of rock-paper-scissors
'''

import glob
import numpy as np
import cv2

# Reading in data and assigning labels
image_list = []
class_list = []
for filename in glob.glob('data/*.png'):
    if "r" in filename:
        class_list.append([1, 0, 0])
    elif "p" in filename:
        class_list.append([0, 1, 0])
    elif "s" in filename:
        class_list.append([0, 0, 1])
    im = cv2.imread(filename)
    # Resizing to 64px by 64px
    image_list.append(cv2.resize(im, (64, 64)))

image_list = np.array(image_list)
class_list = np.array(class_list)

# Splitting dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(image_list, class_list, test_size=0.2, random_state=42)

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D

# CNN Structure
model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#print(model.summary())

# Training the model
MULT = 10
print(X_train.shape)
model.fit(np.repeat(X_train, MULT, axis=0), np.repeat(y_train, MULT, axis=0), validation_data=(X_test, y_test), epochs=50, batch_size=256, verbose=1, shuffle=True)

# Saving our model
model.save("model.h5")
