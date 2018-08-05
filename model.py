from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization, Activation, AveragePooling2D
from keras.callbacks import ModelCheckpoint
import numpy as np


class Classifier:
    self.image_size = 224
    self.channels = 3
    self.classes = 10
