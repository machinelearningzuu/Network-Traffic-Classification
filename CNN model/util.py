import os
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from variables import*

def get_data(csv_path):
    df = pd.read_csv(csv_path)

    labels = df['activity'].values
    Inputs = df.iloc[:,1:].values

    encoder = save_and_load_encoder(labels, csv_path)
    scaler = save_and_load_scalar(Inputs, csv_path)
    assert encoder is not None, "First Load Train data and make Encoder"
    assert scaler is not None, "First Load Train data and make Scalar"

    Inputs = scaler.transform(Inputs)

    if csv_path == train_csv:
        labels = encoder.transform(labels.reshape(-1,1))
        labels = labels.toarray()
        Inputs, labels = shuffle(Inputs, labels)
        return Inputs, labels
    else:
        Inputs = shuffle(Inputs)
        return Inputs

def save_and_load_encoder(labels, csv_path=train_csv):
    if csv_path == train_csv:
        if not os.path.exists(encoder_weights):
            encoder = OneHotEncoder()
            encoder.fit(labels.reshape(-1,1))
            joblib.dump(encoder, encoder_weights)

        elif os.path.exists(encoder_weights):
            encoder = joblib.load(encoder_weights)

    elif os.path.exists(encoder_weights):
        encoder = joblib.load(encoder_weights)

    else:
        return None
    return encoder

def save_and_load_scalar(Inputs, csv_path=train_csv):
    if csv_path == train_csv:
        if not os.path.exists(scalar_weights):
            scaler = StandardScaler()
            scaler.fit(Inputs)
            joblib.dump(scaler, scalar_weights)

        elif os.path.exists(scalar_weights):
            scaler = joblib.load(scalar_weights)

    elif os.path.exists(scalar_weights):
        scaler = joblib.load(scalar_weights)

    else:
        return None
    return scaler

def configure_cnn_inputs(X):
    X = X.reshape(-1, initial_shape[0], initial_shape[1])
    Xnew = np.empty([X.shape[0], input_shape[0], input_shape[1]])
    for i in range(len(X)):
        Xnew[i] = np.resize(X[i],[input_shape[0], input_shape[1]])
        
    Xnew = Xnew.reshape(-1, input_shape[0], input_shape[1], 1)
    return Xnew

def load_data():
    X, Y = get_data(train_csv)
    Xtest  = get_data(test_csv)
    X = configure_cnn_inputs(X)
    Xtest = configure_cnn_inputs(Xtest)
    return X, Y, Xtest