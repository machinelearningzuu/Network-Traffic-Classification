import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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
    # Inputs = scaler.transform(Inputs)

    # pca = PCA(n_components=n_components)
    # pca.fit(Inputs)

    # Inputs = pca.transform(Inputs)

    if not os.path.exists(encoder_weights):
        encoder = OneHotEncoder()
        encoder.fit(labels.reshape(-1,1))
        joblib.dump(encoder, encoder_weights)

    encoder = joblib.load(encoder_weights)

    if Train:
        labels = encoder.transform(labels.reshape(-1,1))
        labels = labels.toarray()

    N, D = Inputs.shape
    Inputs = Inputs.reshape(N, 1, D)
    return encoder, Inputs, labels


def load_data(Train):
    encoder, Inputs, labels = get_data(Train)
    Inputs, labels = shuffle(Inputs, labels)
    return encoder, Inputs, labels





# def get_data_for_one_app(app_name, labels):
#     app_len = len(app_name)
#     app_names = labels.astype(str).str[:app_len].values
#     idx = (app_names == app_name)
#     return idx

# def get_single_app_percentage(trainApp, Pclasses):
#     app_len = len(trainApp)
#     app_names = pd.Series(Pclasses).astype(str).str[:app_len].values
#     idx = (app_names == trainApp)
#     return np.sum(idx)/len(Pclasses) * 100

# def get_app_percentage(Pclasses):
#     for app in TrainApps:
#         percentage = get_single_app_percentage(app, Pclasses)
#         print("{} : {}%".format(FullAppNames[app], round(percentage, 3)))


# def get_apps_data(TrainInputs, TestInputs, Trainlabels, Testlabels):
#     DataDict = {app: '' for app in TrainApps + TestApps}
#     for app in TrainApps:
#         idx = get_data_for_one_app(app, Trainlabels)
#         data = TrainInputs[idx]
#         DataDict[app] = data

#     for app in TestApps:
#         idx = get_data_for_one_app(app, Testlabels)
#         data = TestInputs[idx]
#         DataDict[app] = data

#     return DataDict

# def plot_single(app1, app2, DataDict):
#     data1 = DataDict[app1]
#     data2 = DataDict[app2]

#     min_size = min([len(data1), len(data2)])
#     d1 = data1[:min_size]
#     d2 = data2[:min_size]

#     plt.scatter(d1, d2, c ="red")
#     plt.title(FullAppNames[app1]+" VS "+FullAppNames[app2])
#     plt.xlabel(FullAppNames[app1])
#     plt.ylabel(FullAppNames[app2])
#     plt.savefig(img_corr.format(app1, app2))
#     # plt.show()

# def plot_data(DataDict):
#     DonePlots = []
#     for app1 in TrainApps + TestApps:
#         for app2 in TrainApps + TestApps:
#             if app1 != app2:
#                 if ((app1, app2) not in DonePlots) and ((app2, app1) not in DonePlots):
#                     plot_single(app1, app2, DataDict)
#                     DonePlots.append((app1, app2))

# def visualize_correlation():
#     dfTrain = pd.read_csv(train_csv)
#     dfTest  = pd.read_csv(test_csv)

#     Trainlabels = dfTrain['activity']
#     TrainInputs = dfTrain.iloc[:,1:].values

#     Testlabels  = dfTest['activity']
#     TestInputs  = dfTest.iloc[:,1:].values

#     scaler = StandardScaler()
#     scaler.fit(TrainInputs)

#     TrainInputs = scaler.transform(TrainInputs)
#     TestInputs  = scaler.transform(TestInputs)

#     pca = PCA(n_components=1)
#     pca.fit(TrainInputs)

#     Xtrain = pca.transform(TrainInputs)
#     Xtest  = pca.transform(TestInputs)

#     DataDict = get_apps_data(Xtrain, Xtest, Trainlabels, Testlabels)
#     plot_data(DataDict)

# def app_data():
#     dfTrain = pd.read_csv(train_csv)
#     dfTest  = pd.read_csv(test_csv)

#     Trainlabels = dfTrain['activity']
#     TrainInputs = dfTrain.iloc[:,1:].values

#     Testlabels  = dfTest['activity']
#     TestInputs  = dfTest.iloc[:,1:].values

#     scaler = StandardScaler()
#     scaler.fit(TrainInputs)

#     # TrainInputs = scaler.transform(TrainInputs)
#     # TestInputs  = scaler.transform(TestInputs)

#     pca = PCA(n_components=n_components)
#     pca.fit(TrainInputs)

#     TrainInputs = pca.transform(TrainInputs)
#     TestInputs  = pca.transform(TestInputs)

#     TrainDataDict = {app: '' for app in TrainApps}
#     TestDataDict  = {app: '' for app in TestApps}

#     for app in TrainApps:
#         idx = get_data_for_one_app(app, Trainlabels)
#         data = TrainInputs[idx]
#         TrainDataDict[app] = data

#     for app in TestApps:
#         idx = get_data_for_one_app(app, Testlabels)
#         data = TestInputs[idx]
#         TestDataDict[app] = data

#     return TrainDataDict, TestDataDict

# visualize_correlation()