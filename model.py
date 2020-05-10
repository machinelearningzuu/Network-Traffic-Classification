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
import argparse

'''  Use following command to run the script

                python model.py --data_set=Train

'''

class NetworkTrafficClassifier(object):
    def __init__(self, Train):
        self.Train = Train
        encoder, Inputs, labels = load_data(Train)
        self.X = Inputs
        self.Y = labels
        self.encoder = encoder
        if Train:
            self.num_classes = int(labels.shape[1])
        print("Input Shape : {}".format(self.X.shape))
        print("Label Shape : {}".format(self.Y.shape))

    def classifier(self):
        inputs = Input(shape=(n_features,), name='inputs')
        x = Dense(dense1, activation='tanh', name='dense1')(inputs)
        x = Dense(dense2, activation='tanh', name='dense2')(x)
        x = Dense(dense3, activation='tanh', name='dense3')(x)
        x = Dense(dense4, activation='tanh', name='dense4')(x)
        x = Dense(denset, activation='relu', name='denset')(x)
        x = Dropout(keep_prob)(x)
        outputs = Dense(self.num_classes, activation='softmax', name='output')(x)
        self.model = Model(inputs, outputs)

    @staticmethod
    def acc(y_true,y_pred):
        targ = K.argmax(y_true, axis=-1)
        pred = K.argmax(y_pred, axis=-1)
        correct = K.cast(K.equal(targ, pred), dtype='float32')

        Pmax = K.cast(K.max(y_pred, axis=-1), dtype='float32')
        Pmax = Pmax * correct
        mask = K.cast(K.greater(Pmax, custom_acc), dtype='float32')

        return K.mean(mask)


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

    def evaluation(self):
        Ypred = self.model.predict(self.X)
        Ppred = np.max(Ypred, axis=-1)
        unk = (Ppred <= custom_acc)
        Punk = np.mean(unk) * 100
        print("Unknown {} data Percentage : {}%".format(
                                        "Train" if self.Train else "Test",
                                        round(Punk,3))
                                        )

    def predict_classes(self):
        Ypred = self.model.predict(self.X)

        N = Ypred.shape[0]
        Ppred = np.argmax(Ypred, axis=-1)
        Ponehot = np.zeros((N, train_classes), dtype=np.int64)
        for i in range(N):
           j = Ppred[i]
           Ponehot[i,j] = 1
        Pclasses = self.encoder.inverse_transform(Ponehot)
        return Pclasses

    def predicts(self,X):
        return self.model.predict(X)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    my_parser = argparse.ArgumentParser(description='Model parameters and hyperparameters')

    my_parser.add_argument('--data_set',
                        metavar='train or test',
                        type=str,
                        help='predictions based on required dataset after training the model')

    args = my_parser.parse_args()

    Train = True if args.data_set.lower() == 'train' else False

    model = NetworkTrafficClassifier(Train)
    if os.path.exists(model_weights):
        print("Loading the base model !!!")
        model.load_model(model_weights)
    else:
        print("Training the base model !!!")
        model.train()
    model.evaluation()
    Pclasses = model.predict_classes()
    print(Pclasses)