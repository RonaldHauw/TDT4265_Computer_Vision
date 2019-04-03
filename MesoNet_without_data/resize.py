from classifiers import *
from keras.preprocessing.image import ImageDataGenerator

dataGenerator = ImageDataGenerator(rescale=1./255)

generator = dataGenerator.flow_from_directory(
        'image_data',
        target_size=(256, 256),
        batch_size=1000,
        class_mode='binary',
        subset='training',
        save_to_dir='./gen_images',)


X,y = generator.next()

print()

