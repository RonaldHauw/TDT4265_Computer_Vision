import numpy as np
from classifiers import *
from save_data import *
#from pipeline import *

from keras.preprocessing.image import ImageDataGenerator

"""
*** ------------------------------ MESO4 Classifier --------------------------- ***
"""

# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_DF')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'image_data/train',
        target_size=(256, 256),
        batch_size=100,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        'image_data/test',
        target_size=(256, 256),
        batch_size=100,
        class_mode='binary')


X_train, y_train = train_generator.next()

history = classifier.fit_epochs(X_train,y_train)
print(history.history.keys())

# Save results to files
saver = Save()
saver.save_plots_raw_output(history)


X_test, y_test = test_generator.next()

eval = classifier.get_accuracy_test_evaluate(X_test, y_test)

loss = eval[0]
acc = eval[1]


print('Loss :', loss, '\n  Accuracy:', acc)

"""
*** ------------------------------ MESOInception-4 Classifier --------------------------- ***
"""

# 1 - Load the model and its pretrained weights
classifier = MesoInception4()
classifier.load('weights/MesoInception_DF')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'image_data/train',
        target_size=(256, 256),
        batch_size=100,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        'image_data/test',
        target_size=(256, 256),
        batch_size=100,
        class_mode='binary')


X_train, y_train = train_generator.next()

history = classifier.fit_epochs(X_train,y_train)
print(history.history.keys())

# Save results to files
saver = Save()
saver.save_plots_raw_output(history)


X_test, y_test = test_generator.next()

eval = classifier.get_accuracy_test_evaluate(X_test, y_test)

loss = eval[0]
acc = eval[1]


print('Loss :', loss, '\n  Accuracy:', acc)
