import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, LSTM
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K
import logging
logging.getLogger('tensorflow').disabled = True

from variables import*
from util import load_data

'''  Use following command to run the script

                python -W ignore model.py

'''

class NetworkTrafficClassifier(object):
    def __init__(self, Train):
        encoder, Inputs, labels = load_data(Train)
        self.X = Inputs
        self.Y = labels
        self.encoder = encoder
        self.num_classes = int(labels.shape[1])
        print("Dataset has {} Classes".format(self.num_classes))

    def classifier(self):
        inputs = Input(shape=(n_features,), name='inputs')
        x = Dense(dense1, activation='relu', name='dense1')(inputs)
        x = Dense(dense2, activation='relu', name='dense2')(x)
        x = Dense(dense3, activation='relu', name='dense3')(x)
        x = Dense(dense4, activation='relu', name='dense4')(x)
        x = Dropout(keep_prob)(x)
        outputs = Dense(self.num_classes, activation='softmax', name='output')(x)
        self.model = Model(inputs, outputs)

    @staticmethod
    def acc(y_true,y_pred):
        print(y_pred.shape)
        n_total = K.cast(K.shape(y_pred)[0], dtype='float32')
        Plabels = K.cast(K.argmax(y_pred, axis=-1), dtype='float32') # label array
        print(K.shape(Plabels))
        print(K.shape(y_true))
        correct_idx = K.cast(K.equal(y_true, Plabels), dtype='float32')
        n_correct = K.sum(correct_idx)
        # Pmax = K.cast(K.max(y_pred, axis=-1), dtype='float32')
        # Pmax = Pmax * correct_idx
        # mask = K.cast(K.greater(Pmax, custom_acc), dtype='float32')

        # n_correct = K.sum(mask)
        return n_correct/n_total


    def train(self):
        self.classifier()
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            # metrics=['accuracy'],
            metrics=[NetworkTrafficClassifier.acc]
        )
        self.history = self.model.fit(
                            self.X,
                            self.Y,
                            batch_size=batch_size,
                            epochs=num_epoches,
                            validation_split=validation_split
                            )
        self.save_model(model_weights)

    def load_model(self, model_weights):
        loaded_model = load_model(model_weights)
        loaded_model.compile(
                        loss='categorical_crossentropy',
                        optimizer='adam',
                        # metrics=['accuracy'],/
                        metrics=[NetworkTrafficClassifier.acc]
                        )
        self.model = loaded_model

    def save_model(self ,model_weights):
        self.model.save(model_weights)

    def evaluation(self,X,Y):
        loss, accuracy = self.model.evaluate(X, Y, batch_size=batch_size)
        Ypred = self.model.predict(X)
        P = np.argmax(Ypred, axis=1)
        print("Test Loss : ", loss)
        print("Test Accuracy : ", accuracy)
        print("Predicted Classes : \n", P)

    def predicts(self,X):
        return self.model.predict(X)

if __name__ == "__main__":
    Train = False if os.path.exists(model_weights) else True
    model = NetworkTrafficClassifier(Train)
    if os.path.exists(model_weights):
        print("Loading the base model !!!")
        model.load_model(model_weights)
    else:
        print("Training the base model !!!")
        model.train()
    encoder, X, Y = load_data(True)
    P = model.predicts(X)
    print(P.shape)