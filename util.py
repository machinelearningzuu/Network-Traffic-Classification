import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from variables import*

def get_data():
    Input = []
    Outputs = []
    frame_counts = []
    for label,class_name in enumerate(os.listdir(main_dir)):
        class_dir = os.path.join(main_dir,class_name)
        frame_count, X, Y = get_data_class(label, class_dir)
        Input.extend(X)
        Outputs.extend(Y)
        frame_counts.extend(frame_count)
    Pad_inputs = pad_data(Input)
    Outputs = np.array(Outputs)
    # return frame_counts,Pad_inputs,Outputs
    return Pad_inputs,Outputs

def get_data_class(label, class_dir):
    csv_files = os.listdir(class_dir)
    X = []
    Y = []
    frame_count = []
    for csv_file in csv_files:
        csv_path = os.path.join(class_dir,csv_file)
        df = pd.read_csv(csv_path)
        data = df[['frame.time_delta_displayed','frame.len']]
        data = data.copy().dropna(axis = 0, how ='any')
        if len(data) > 0:
            x = data.values
            X.append(x)
            Y.append(label)
            frame_count.append(len(x))
    return frame_count, X, Y

def pad_data(Input):
    N = len(Input)
    Pad_inputs = np.empty([N, frame_count_threshold, n_features])
    for idx,x in enumerate(Input):
        if len(x) >= frame_count_threshold:
            new_x = x[:frame_count_threshold,]
        else:
            new_x = np.zeros((frame_count_threshold, n_features))
            len_x = len(x)
            new_x[-len_x:, ] = x
        Pad_inputs[idx,:,:] = new_x
    return Pad_inputs

def normalize_data(x):
    x_min1 = x[:,:,0].min()
    x_max1 = x[:,:,0].max()
    x[:,:,0] = (x[:,:,0] - x_min1)/(x_max1 - x_min1)

    x_min2 = x[:,:,1].min()
    x_max2 = x[:,:,1].max()
    x[:,:,1] = (x[:,:,1] - x_min2)/(x_max2 - x_min2)

    return x

def load_data():
    Pad_inputs,Outputs = get_data()
    Pad_inputs,Outputs = shuffle(Pad_inputs,Outputs)
    Pad_inputs = normalize_data(Pad_inputs)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
                                                    Pad_inputs,
                                                    Outputs,
                                                    test_size=validation_split,
                                                    random_state=seed
                                                    )

    return Xtrain, Xtest, Ytrain, Ytest