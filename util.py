import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from variables import*

import joblib

def get_data(Train):
    csv_path = train_csv if Train else test_csv
    df = pd.read_csv(csv_path)
    class_names = df['activity'].values
    unique_classes = set(class_names)

    encoder = LabelEncoder()
    encoder.fit(class_names)
    labels = encoder.transform(class_names)

    Inputs = df.iloc[:,1:].values
    scaler = StandardScaler()
    scaler.fit(Inputs)
    Inputs = scaler.transform(Inputs)

    return encoder, unique_classes, Inputs, labels


def load_data(Train=True):
    encoder, unique_classes, Inputs, labels = get_data(Train)
    Inputs, labels = shuffle(Inputs, labels)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
                                                    Inputs,
                                                    labels,
                                                    test_size=validation_split,
                                                    random_state=seed
                                                    )

    # if not os.path.exists(pca_weights):
    #     pca = PCA(n_components=n_features)
    #     pca.fit(Xtrain)
    #     joblib.dump(pca, pca_weights)

    # pca = joblib.load(pca_weights)
    # Xtrain = pca.transform(Xtrain)
    # Xtest  = pca.transform(Xtest)

    return encoder, unique_classes, Xtrain, Xtest, Ytrain, Ytest
