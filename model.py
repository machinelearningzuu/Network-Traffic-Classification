import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Dropout, Embedding, LSTM
from keras.models import model_from_json
from keras.models import Model

import keras.backend as K
import logging
logging.getLogger('tensorflow').disabled = True

from variables import*
from util import load_data

'''  Use following command to run the script

                python -W ignore model.py

'''

class NetworkTrafficClassifier(object):
    def __init__(self):
        encoder, unique_classes, Xtrain, Xtest, Ytrain, Ytest = load_data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.encoder = encoder
        self.unique_classes = unique_classes
        self.num_classes = len(unique_classes)
        print("Dataset has {} Classes".format(self.num_classes))

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

    def load_model(self, model_path, model_weights):
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

    def evaluation(self,X,Y):
        loss, accuracy = self.model.evaluate(X, Y, batch_size=batch_size)
        Ypred = self.model.predict(X)
        P = np.argmax(Ypred, axis=1)
        Pclasses = self.encoder.inverse_transform(P)
        self.plot_confusion(Y, P)
        print("Test Loss : ", loss)
        print("Test Accuracy : ", accuracy)
        print("Predicted Classes : \n", Pclasses)

    def plot_confusion(self,Y, P):
        classes = self.encoder.transform(list(self.unique_classes))
        confusion_matrix = tf.math.confusion_matrix(labels=Y, predictions=P).numpy()
        # confusion_matrix = np.around(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2)
        confusion_matrix_df = pd.DataFrame(
                                confusion_matrix,
                                index = classes,
                                columns = classes
                                )
        figure = plt.figure(figsize=(10, 10))
        sns.heatmap(confusion_matrix_df, annot=True,cmap= plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True classes')
        plt.xlabel('Predicted classes')
        plt.savefig(confustion_img)

    def predicts(self,X):
        return self.model.predict(X)

if __name__ == "__main__":
    model = NetworkTrafficClassifier()
    _, _, Xtrain, Xtest, Ytrain, Ytest = load_data(False)
    if os.path.exists(model_path) and os.path.exists(model_weights):
        print("Loading the base model !!!")
        model.load_model(model_path,model_weights)
    else:
        print("Training the base model !!!")
        model.train()