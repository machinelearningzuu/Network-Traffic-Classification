import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from variables import*

def get_data():
    csv_path = base_csv if base_model else tl_path
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


def load_data():
    encoder, unique_classes, Inputs, labels = get_data()
    Inputs, labels = shuffle(Inputs, labels)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
                                                    Inputs,
                                                    labels,
                                                    test_size=validation_split,
                                                    random_state=seed
                                                    )

    return encoder, unique_classes, Xtrain, Xtest, Ytrain, Ytest
