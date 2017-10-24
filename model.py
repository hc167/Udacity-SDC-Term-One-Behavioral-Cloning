import numpy as np
from keras.models import Sequential, Model
from keras.layers import Lambda, Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import Cropping2D
import cv2
import os
import csv
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam
import h5py
import random

samples = []
with open('../driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.3)
validation_samples, test_samples = train_test_split(validation_samples, test_size=0.3)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                pathstring = batch_sample[0].replace("\\", "/")
                name = '../IMG/'+pathstring.split('/')[-1]                    
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


adjustdegree = 0.4
def generatorTrain(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                    rand = random.randrange(0,4)
                    ind = 0
                    if rand == 0:
                        ind = 0
                        center_angle = float(batch_sample[3])
                    elif rand == 1:
                        ind = 1
                        center_angle = float(batch_sample[3]) + adjustdegree
                    else:
                        ind = 2
                        center_angle = float(batch_sample[3]) - adjustdegree

                    pathstring = batch_sample[ind].replace("\\", "/")
                    name = '../IMG/'+pathstring.split('/')[-1]                    
                    center_image = cv2.imread(name)
                    
                    if random.randrange(0,2) == 0:
                        center_image = np.fliplr(center_image)
                        center_angle = -center_angle
                    
                    images.append(center_image)
                    angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generatorTrain(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
test_generator = generator(test_samples, batch_size=32)

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0, 0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
          
# Preprocess incoming data, centered around zero with small standard deviation 
model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, steps_per_epoch= len(train_samples)*3,
                    validation_data=validation_generator, validation_steps=len(validation_samples), epochs=1)

scores = model.evaluate_generator(test_generator, len(test_samples))
print("Test data loss = ", scores)

model.save('model.h5')