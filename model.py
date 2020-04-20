import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

from variables import*
from util import load_data
from keras.layers import Dense, Input, Dropout
from keras.models import model_from_json
from keras.models import Model

import logging
logging.getLogger('tensorflow').disabled = True

class NetworkTrafficClassifier(object):
    def __init__(self):
        num_classes, Xtrain, Xtest, Ytrain, Ytest = load_data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.num_classes = num_classes

    def classifier(self):
        inputs = Input(shape=(n_features,), name='inputs')
        x = Dense(dense1, activation='tanh', name='dense1')(inputs)
        x = Dense(dense2, activation='tanh', name='dense2')(x)
        x = Dense(dense3, activation='tanh', name='dense3')(x)
        x = Dense(dense4, activation='tanh', name='dense4')(x)
        x = Dropout(keep_prob)(x)
        outputs = Dense(self.num_classes, activation='softmax', name='output')(x)
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
        self.save_model(model_path,model_weights)

    def load_model(self):
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weights)
        loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = loaded_model

    def save_model(self,model_path,model_weights):
        model_json = self.model.to_json()
        with open(model_path, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_weights)

    def transfer_learning(self):
        self.load_model()
        inputs = Input(shape=(n_features,), name='inputs')
        x = inputs
        for layer in self.model.layers[1:-1]:
            x = layer(x)
            layer.trainable = False
        x = Dense(denset, activation='relu', name='denset')(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs, outputs)

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
        self.save_model(tl_model_path,tl_model_weights)


if __name__ == "__main__":
    model = NetworkTrafficClassifier()
    if os.path.exists(model_path) and os.path.exists(model_weights):
        if base_model:
            print("Loaing the base model !")
            model.load_model()
        else:
            print("Transfer Learning !")
            model.transfer_learning()
    else:
        print("Training the base model !")
        model.train()