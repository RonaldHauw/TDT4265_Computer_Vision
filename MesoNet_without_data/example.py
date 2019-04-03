import numpy as np
from classifiers import *
#from pipeline import *

from keras.preprocessing.image import ImageDataGenerator

# 1 - Load the model and its pretrained weights
classifier = Meso4()
classifier.load('weights/Meso4_DF')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)

dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'image_data',
        target_size=(256, 256),
        batch_size=1000,
        class_mode='binary',
        subset='training')

# 3 - Predict
X, y = generator.next()

# Just runs the whole dataset through the pre-trained MESO4 model and retuns the loss and accuracy
eval = classifier.get_accuracy(X,y)

loss = eval[0]
acc = eval[1]


print('Loss :', loss, '\n  Accuracy:', acc)

# 4 - Prediction for a video dataset

#classifier.load('weights/Meso4_F2F')

#predictions = compute_accuracy(classifier, 'test_videos')
#for video_name in predictions:
#    print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])
