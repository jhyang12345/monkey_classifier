from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization, Activation, AveragePooling2D
from keras.callbacks import ModelCheckpoint
import numpy as np


class Classifier:
    def __init__(self):
        self.image_size = 224
        self.channels = 3
        self.classes = 10

    def build_model(self):
        c_input = Input(shape=(self.image_size, self.image_size, self.channels))

        model = VGG19(include_top = False, weights='imagenet', input_shape=(self.image_size, self.image_size, self.channels))

        for layer in model.layers:
            layer.trainable = False

        x = model.output

        # x = Flatten()(x)
        # x = Dense(256, activation='relu')(x)
        # x = Dropout(0.3)(x)
        # x = Dense(256, activation='relu')(x)

        x = Conv2D(filters=1024, kernel_size=4, padding='same', activation='relu')(x)
        # x = Conv2D(filters=512, kernel_size=4, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)
        #
        x = Dropout(0.5)(x)

        x = Conv2D(filters=1024, kernel_size=4, padding='same', activation='relu')(x)
        # x = Conv2D(filters=512, kernel_size=4, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)
        #
        x = Dropout(0.5)(x)

        # x = Conv2D(filters=512, kernel_size=4, padding='same', activation='relu')(x)
        # x = Conv2D(filters=512, kernel_size=4, padding='same', activation='relu')(x)
        # x = GlobalAveragePooling2D()(x)
        # x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        logits = Dense(self.classes)(x)
        predictions = Activation("softmax")(logits)

        model = Model(model.input, predictions)
        model_output = model(c_input)
        model.summary()

        model.compile(loss='categorical_crossentropy', optimizer='adam',
                    metrics=['accuracy'])

        return model
