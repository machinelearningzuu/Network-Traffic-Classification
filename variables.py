import os

seed=42
n_features = 48
kernal_size = 3
pool_size = 2
output_dim = 256
num_classes = 74
batch_size = 128
num_epoches = 100
validation_split = 0.2

#data paths and model weights
big_csv = os.path.join(os.getcwd(), 'AllData6Apps0.5s.csv')

model_path =  os.path.join(os.getcwd(), 'network_classifier.json')
model_weights =  os.path.join(os.getcwd(), 'network_classifier.h5')