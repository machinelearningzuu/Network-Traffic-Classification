import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import operator
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPool2D, Flatten

logging.getLogger('tensorflow').disabled = True

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\nNum GPUs Available : {}".format(len(physical_devices)))
print("Tensorflow version : {}\n".format(tf.__version__))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from variables import*
from util import load_data

class TrafficClassifier(object):
    def __init__(self):
        X, Y = load_data()
        self.X = X 
        self.Y = Y
        print("Input Shape : {}".format(self.X.shape))
        print("Label Shape : {}".format(self.Y.shape))

    def classifier(self):
        n_classes = len(drop_enc.categories_[0])
        inputs = Input(shape=input_shape)
        x = Conv2D(256, (3,3), activation='relu')(inputs)
        x = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Conv2D(128, (3,3), activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Conv2D(64, (3,3), activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(keep_prob)(x)
        outputs = Dense(n_classes, activation='softmax')(x)
        self.model = Model(inputs, outputs)
        self.model.summary()

    @staticmethod
    def network_acc(custom_acc):
        def acc(y_true,y_pred):
            targ = K.argmax(y_true, axis=-1)
            pred = K.argmax(y_pred, axis=-1)

            correct = K.cast(K.equal(targ, pred), dtype='float32')
            Pmax = K.cast(K.max(y_pred, axis=-1), dtype='float32')

            Pmax = Pmax * correct
            mask = K.cast(K.greater(Pmax, custom_acc), dtype='float32')

            return K.mean(mask)
        return acc


    def train(self):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate),
            metrics=['accuracy']
            # metrics=[
            #     TrafficClassifier.network_acc(
            #                             custom_acc=custom_acc
            #                                  )]
        )
        self.history = self.model.fit(
                            self.X,
                            self.Y,
                            batch_size=batch_size,
                            epochs=num_epoches,
                            validation_split=validation_split
                            )

    def load_model(self):
        loaded_model = load_model(model_weights)
        loaded_model.compile(
                        loss='categorical_crossentropy',
                        optimizer=Adam(learning_rate),
                        metrics=['accuracy']
                        # metrics=[
                        #     TrafficClassifier.network_acc(
                        #                             custom_acc=custom_acc
                        #                                 )]
                        )
        self.model = loaded_model

    def save_model(self):
        self.model.save(model_weights)

    def evaluation(self, X):
        Ypred = self.model.predict(X)
        Ppred = np.max(Ypred, axis=-1)
        unk = (Ppred <= custom_acc)
        Punk = np.mean(unk) * 100
        return Punk

    def run(self):
        if os.path.exists(model_weights):
            print("Loading the CNN !!!")
            self.load_model()
        else:
            print("Training the CNN !!!")
            self.classifier()
            self.train()
            self.save_model()

if __name__ == "__main__":
    model = TrafficClassifier()
    model.run()
