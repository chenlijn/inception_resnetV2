import os
import numpy
import cv2
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class Inception_Res_NetV2(object):

    def __init__(self, class_num):
        self._class_num = class_num
        self._input_shape = (299, 299, 3)
        self._input_size = (299, 299)
        self._model = InceptionResNetV2(include_top=True, weights=None, classes=self._class_num)

    def train_network(self, src_dir, model_path, train_image_num, test_image_num, batch_size, epoch_num, lr):
        sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9)
        self._model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

        train_datagen = ImageDataGenerator(rescale = 1./255)
        test_datagen = ImageDataGenerator(rescale = 1./255)

        train_generator = train_datagen.flow_from_directory(src_dir+'train/', 
                                                            target_size = self._input_size,
                                                            batch_size = 32, 
                                                            class_mode = 'categorical')

        test_generator = test_datagen.flow_from_directory(src_dir+'val/', 
                                                            target_size = self._input_size,
                                                            batch_size = 32, 
                                                            class_mode = 'categorical')

        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True)
        early_stop = EarlyStopping(monitor = 'val_loss', verbose=1, patience=20)
        reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.1, pathience=10, verbose=1)
        self._model.fit_generator(train_generator, 
                                  steps_per_epoch=train_image_num//batch_size,
                                  epochs = epoch_num,
                                  validation_data = test_generator,
                                  validation_steps = test_image_num//batch_size,
                                  callbacks = [checkpoint, early_stop, reducer])


if __name__=='__main__':

    src_dir = '/root/mount_out/data/left_right_disc_data/train_val_data/'
    model_path = 'model/weights-{epoch:04d}-{val_loss:.3f}.hdf5'
    train_image_num = 821
    test_image_num = 205
    batch_size = 32
    epoch_num = 200
    lr = 0.01

    model = Inception_Res_NetV2(2)
    model.train_network(src_dir, model_path, train_image_num, test_image_num, batch_size, epoch_num, lr)


