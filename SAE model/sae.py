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
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, LSTM, BatchNormalization, Conv1D, GRU, concatenate
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

import tensorflow.keras.backend as K
import logging
logging.getLogger('tensorflow').disabled = True

from variables import*
from util import*
import argparse
import operator
from collections import Counter

'''  Use following command to run the script

                python model.py --data_set=Train

'''

class NetworkTrafficClassifier(object):
    def __init__(self, Train):
        self.Train = Train
        encoder, Inputs_sae, Inputs_conv, labels = load_data(Train)
        self.X_sae = Inputs_sae
        self.X_conv = Inputs_conv
        self.Y = labels
        self.encoder = encoder
        if Train:
            self.num_classes = int(labels.shape[1])
        print("CNN Input Shape : {}".format(self.X_conv.shape))
        print("Autoencoder Input Shape : {}".format(self.X_sae.shape))
        print("Label Shape : {}".format(self.Y.shape))

    def autoencoder(self):
        if os.path.exists(autoencoder_weights):
            print("Autoencoder model Loading !!!")
            self.autoencoder_model = self.load_model(autoencoder_weights)
            self.autoencoder_model.compile(
                                    optimizer=Adam(learning_rate),
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy']
                                    )
        else:
            print("Autoencoder model Training !!!")
            inputs = Input(shape=(n_features,))
            x = Dense(dim1, activation='relu', name="dense1")(inputs)
            x = Dense(dim2, activation='relu', name="dense2")(x)
            x = Dense(dim2, activation='relu', name="dense3")(x)
            x = Dense(dim3, activation='relu', name="dense4")(x)
            x = Dense(dim2, activation='relu', name="dense5")(x)
            x = Dense(dim2, activation='relu', name="dense6")(x)
            x = Dense(dim1, activation='relu', name="dense7")(x)
            output = Dense(n_features, activation='relu', name="output_sae")(x)

            self.autoencoder_model = Model(inputs=inputs,
                                      outputs=output)

            self.autoencoder_model.compile(
                                    optimizer=Adam(learning_rate),
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy']
                                    )


            self.history_auctoencoder = self.autoencoder_model.fit( self.X_sae,
                                                                    self.X_sae,
                                                                    epochs=num_epoches,
                                                                    batch_size=batch_size,
                                                                    validation_split=validation_split,
                                                                    verbose = 1
                                                            )

            self.save_model(self.autoencoder_model, autoencoder_weights)

    def SAE_model(self):
        if os.path.exists(sae_weights):
            print("\nStack Autoencoder model Loading !!!")
            self.sae_model = self.load_model(sae_weights)
            self.sae_model.compile(
                                optimizer=Adam(learning_rate),
                                loss='categorical_crossentropy',
                                metrics=['accuracy']
                                )

        else:
            self.autoencoder()
            self.sae_model = self.autoencoder_model
            print("\nStack autoencoder model Training !!!")

            for i in range(len(self.sae_model.layers[1:])):
                for j,layer in enumerate(self.sae_model.layers[1:]):
                    if i != j:
                        layer.trainable = False
                    else:
                        print("\n------{} finetuning-------".format(layer.name))

                self.sae_model.compile(
                        optimizer=Adam(learning_rate),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                        )
                self.sae_model.fit( self.X_sae,
                                    self.X_sae,
                                    epochs=num_epoches,
                                    batch_size=batch_size,
                                    validation_split=validation_split,
                                    verbose = 1
                                    )
            self.save_model(self.sae_model, sae_weights)

    def CNN_1D(self):
        if os.path.exists(model_weights):
            print("\nLoading the 1D CNN model !!!")
            self.model = self.load_model(model_weights)
            self.model.compile(
                                loss='categorical_crossentropy',
                                optimizer=Adam(learning_rate),
                                metrics=['accuracy']
                            )
        else:
            print("\nTraining the 1D CNN model !!!")
            inputs = Input(shape=(1, n_features))
            x = Conv1D(dense1, kernal_size, activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = Conv1D(dense2, kernal_size, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Conv1D(dense2, kernal_size, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Conv1D(dense2, kernal_size, activation='relu')(x)
            x = BatchNormalization()(x)
            x = GRU(dense1)(x)
            x = Dropout(keep_prob)(x)
            x = Dense(dense1, activation='relu')(x)
            x = Dense(dense3, activation='relu')(x)
            x = Dense(dense3, activation='relu')(x)
            x = Dropout(keep_prob)(x)
            output = Dense(train_classes, activation='softmax', name="output_cnn")(x)

            self.model = Model( inputs=inputs,
                                outputs=output)
            self.model.compile(
                                loss='categorical_crossentropy',
                                optimizer=Adam(learning_rate),
                                metrics=['accuracy']
                            )

            self.history = self.model.fit(
                                self.X_conv,
                                self.Y,
                                batch_size=batch_size,
                                epochs=num_epoches,
                                validation_split=validation_split
                                )

            self.save_model(self.model, model_weights)

    def Final_model(self):
        self.CNN_1D()
        self.SAE_model()

        if os.path.exists(final_model_weights):
            print("\nLoading the Final model !!!")
            self.final_model = self.load_model(final_model_weights)
            self.final_model.compile(
                                loss='categorical_crossentropy',
                                optimizer=Adam(learning_rate),
                                metrics=['accuracy']
                            )
        else:
            print("\nTraining the Final model !!!")
            output_cnn = self.model.get_layer('output_cnn').output
            output_sae = self.sae_model.get_layer('output_sae').output
            merged = concatenate([output_cnn, output_sae])
            x = Dense(dense2, activation='relu')(merged)
            x = Dense(dense1, activation='relu')(x)
            x = Dense(dense3, activation='relu')(x)
            x = Dense(dense3, activation='relu')(x)
            final_output = Dense(train_classes, activation='relu')(x)


            self.final_model = Model(   inputs=merged,
                                        outputs=final_output)
            self.final_model.compile(
                                loss='categorical_crossentropy',
                                optimizer=Adam(learning_rate),
                                metrics=['accuracy']
                            )

            self.final_model.fit(
                                self.X_conv,
                                self.Y,
                                batch_size=batch_size,
                                epochs=num_epoches,
                                validation_split=validation_split
                                )

            self.save_model(self.final_model, final_model_weights)

    def plot_metrics(self):
        plot_steps = num_epoches // plot_step

        loss_train = self.history.history['loss']
        loss_val = self.history.history['val_loss']
        loss_train = [loss for i,loss in enumerate(loss_train) if (i+1)%plot_step == 0]
        loss_val = [loss for i,loss in enumerate(loss_val) if (i+1)%plot_step == 0]
        plt.plot(np.arange(1,plot_steps+1), loss_train, 'r', label='Training loss')
        plt.plot(np.arange(1,plot_steps+1), loss_val, 'b', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs // {}'.format(plot_step))
        plt.ylabel('Loss')
        plt.savefig(loss_img)
        plt.legend()
        plt.show()

        acc_train = self.history.history['acc']
        acc_val = self.history.history['val_acc']
        acc_train = [acc for i,acc in enumerate(acc_train) if (i+1)%plot_step == 0]
        acc_val = [acc for i,acc in enumerate(acc_val) if (i+1)%plot_step == 0]
        plt.plot(np.arange(1,plot_steps+1), acc_train, 'r', label='Training Accuracy')
        plt.plot(np.arange(1,plot_steps+1), acc_val, 'b', label='validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs // {}'.format(plot_step))
        plt.ylabel('Accuracy')
        plt.savefig(acc_img)
        plt.legend()
        plt.show()

    def load_model(self, weights):
        loaded_model = load_model(weights)
        return loaded_model

    def save_model(self , model, weights):
        model.save(weights)

    def run(self):
        self.CNN_1D()
        self.SAE_model()

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
    model.run()