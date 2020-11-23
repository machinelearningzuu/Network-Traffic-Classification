import os
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
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization

logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\nNum GPUs Available : {}".format(len(physical_devices)))
print("Tensorflow version : {}\n".format(tf.__version__))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from variables import*
from util import load_data

class TrafficClassifier(object):
    def __init__(self):
        X, Y, Xtest = load_data()
        self.X = X 
        self.Y = Y
        self.Xtest = Xtest
        print("Input Shape : {}".format(self.X.shape))
        print("Label Shape : {}".format(self.Y.shape))

    def classifier(self):
        inputs = Input(shape=(n_features,))
        x = Dense(dense1, activation='relu')(inputs)
        x = Dense(dense1, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(dense2, activation='relu')(x)
        x = Dense(dense2, activation='relu')(x)
        x = Dense(dense2, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(dense3, activation='relu')(x)
        x = Dense(dense3, activation='relu')(x)
        x = Dense(dense3, activation='relu')(x)
        x = Dense(dense3, activation='relu')(x)
        x = Dropout(keep_prob)(x)
        outputs = Dense(n_classes, activation='softmax')(x)
        self.model = Model(inputs, outputs)

    @staticmethod
    def network_acc(custom_acc):
        def acc(y_true,y_pred):
            targ = K.argmax(y_true, axis=-1)
            pred = K.argmax(y_pred, axis=-1)

            correct = K.cast(K.equal(targ, pred), dtype='float32')
            Pmax = K.cast(K.max(y_pred, axis=-1), dtype='float32')

            assert tf.math.reduce_all(
                            tf.math.equal(
                                        K.shape(correct), 
                                        K.shape(Pmax)
                                         )
                                     ),"Invalid Dimension in custom accuracy"
            Pmax = tf.multiply(Pmax , correct)
            mask = K.cast(K.greater(Pmax, custom_acc), dtype='float32')

            return K.mean(mask)
        return acc


    def train(self):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate),
            # metrics=['accuracy'],
            metrics=[
                TrafficClassifier.network_acc(
                                        custom_acc=custom_acc
                                             )]
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
                        # metrics=['accuracy'],
                        metrics=[
                            TrafficClassifier.network_acc(
                                                    custom_acc=custom_acc
                                                        )]
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

    def unknown_evaluation(self):
        print("-----Unknown percentage------")
        print("Train Data : {}%".format(self.evaluation(self.X)))
        print("Test Data : {}%".format(self.evaluation(self.Xtest)))

    def predict_classes(self):
        Ypred = self.model.predict(self.X)

        N = Ypred.shape[0]
        Ppred = np.argmax(Ypred, axis=-1)
        Ponehot = np.zeros((N, n_classes), dtype=np.int64)
        for i in range(N):
           j = Ppred[i]
           Ponehot[i,j] = 1
        Pclasses = self.encoder.inverse_transform(Ponehot).reshape(-1,)
        class_count = dict(Counter(Pclasses.tolist()))
        class_count = sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
        for label, value in class_count:
            fraction = (value/N)*100
            fraction = round(fraction, 3)
            print("{} : {}%".format(label,fraction))

    def predicts(self,X):
        return self.model.predict(X)

    # def predict_distribution():

    def run(self):
        if os.path.exists(model_weights):
            print("Loading the model !!!")
            self.load_model()
        else:
            print("Training the model !!!")
            self.classifier()
            self.train()
            self.save_model()
        self.unknown_evaluation()

if __name__ == "__main__":
    model = TrafficClassifier()
    model.run()