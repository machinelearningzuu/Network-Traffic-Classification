import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from variables import*

import joblib

def get_data(Train):
    csv_path = train_csv if Train else test_csv
    df = pd.read_csv(csv_path)

    labels = df['activity'].values
    Inputs = df.iloc[:,1:].values

    if not os.path.exists(scalar_weights):
        scaler = StandardScaler()
        scaler.fit(Inputs)
        joblib.dump(scaler, scalar_weights)

    scaler = joblib.load(scalar_weights)
    Inputs = scaler.transform(Inputs)

    if not os.path.exists(encoder_weights):
        encoder = OneHotEncoder()
        encoder.fit(labels.reshape(-1,1))
        joblib.dump(encoder, encoder_weights)

    encoder = joblib.load(encoder_weights)

    if Train:
        labels = encoder.transform(labels.reshape(-1,1))
        labels = labels.toarray()
    return encoder, Inputs, labels


def load_data(Train):
    encoder, Inputs, labels = get_data(Train)
    Inputs, labels = shuffle(Inputs, labels)
    return encoder, Inputs, labels
