# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 23:26:23 2023

@author: Dell
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()

# 1. Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (128, 128, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 2. Convolution katmanı
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# YSA
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))

# CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# CNN ve resimler

from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2, 
                                   horizontal_flip = True) 

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory("C:\\Users\\Dell\\Desktop\\Covid 19 CNN\\xray_dataset_covid19\\train",
                                                 target_size = (128, 128),
                                                 color_mode = "grayscale",
                                                 batch_size = 1,
                                                 class_mode = 'binary') 

test_set = test_datagen.flow_from_directory("C:\\Users\\Dell\\Desktop\\Covid 19 CNN\\xray_dataset_covid19\\test",
                                            target_size = (128, 128),
                                            color_mode = "grayscale",
                                            batch_size = 1,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,                         
                         epochs = 15,
                         validation_data = test_set)

import numpy as np
import pandas as pd


test_set.reset()
pred=classifier.predict_generator(test_set,verbose=1) 
pred[pred > .5] = 1
pred[pred <= .5] = 0

print('prediction gecti')

test_labels = []

for i in range(0,int(40)):
    test_labels.extend(np.array(test_set[i][1]))
    
print('test_labels')
print(test_labels)

dosyaisimleri = test_set.filenames
sonuc = pd.DataFrame()
sonuc['dosyaisimleri']= dosyaisimleri
sonuc['tahminler'] = pred
sonuc['test'] = test_labels   

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, pred)
print (cm)