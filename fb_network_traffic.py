import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

from variables import*
from util import load_data
from keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, LSTM, Dropout
from keras.models import model_from_json
from keras.models import Model

import logging
logging.getLogger('tensorflow').disabled = True

class NetworkTrafficClassifier(object):
    def __init__(self):
        Xtrain, Xtest, Ytrain, Ytest = load_data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest

    def classifier(self):
        inputs = Input(shape=(n_features,), name='inputs')
        x = Dense(512, activation='tanh')(inputs)
        x = Dense(512, activation='tanh')(x)
        x = Dense(128, activation='tanh')(x)
        x = Dense(128, activation='tanh')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        self.model = Model(inputs, outputs)

    def train(self):
        self.classifier()
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        self.history = self.model.fit(
                            self.Xtrain,
                            self.Ytrain,
                            batch_size=batch_size,
                            epochs=num_epoches,
                            validation_data = [self.Xtest, self.Ytest],
                            )
        self.save_model()

    def load_model(self):
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weights)
        loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = loaded_model

    def save_model(self):
        model_json = self.model.to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_weights)

    def transfer_learning(self):
        self.load_model()
        for i in self.model.layers:
            print(i)


if __name__ == "__main__":
    model = NetworkTrafficClassifier()
    if os.path.exists(model_path) and os.path.exists(model_weights):
        # model.load_model()
        model.transfer_learning()
    else:
        model.train()
