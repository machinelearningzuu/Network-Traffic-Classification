import os

base_model = True
seed=42

dense0 = 1024
dense1 = 512
dense2 = 512
dense3 = 64
dense4 = 64
denset = 64
keep_prob = 0.7
n_features = 48

batch_size = 128
num_epoches = 100
validation_split = 0.1

custom_acc = 0.97

#data paths and model weights
pca_weights = os.path.join(os.getcwd(), 'Weights/pca.pickle')
model_weights =  os.path.join(os.getcwd(), 'Weights/model.h5')

train_csv = os.path.join(os.getcwd(), 'Datasets 0.5s/Train0.5s.csv')
test_csv = os.path.join(os.getcwd(), 'Datasets 0.5s/Test0.5s.csv')
