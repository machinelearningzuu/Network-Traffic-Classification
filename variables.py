import os

base_model = True
seed=42
dense1 = 512
dense2 = 512
dense3 = 128
dense4 = 128
denset = 32
keep_prob = 0.3
output_dim = 256
n_features = 48
batch_size = 128
num_epoches = 10
validation_split = 0.2

#data paths and model weights
# tl - transfer_learning
base_csv = os.path.join(os.getcwd(), 'DataFiles/FGISY0.5s.csv')
tl_path = os.path.join(os.getcwd(), 'DataFiles/M0.5s.csv')
base_model_path =  os.path.join(os.getcwd(), 'DataFiles/base_model.json')
base_model_weights =  os.path.join(os.getcwd(), 'DataFiles/base_model.h5')
tl_model_path =  os.path.join(os.getcwd(), 'DataFiles/transfer_learning.json')
tl_model_weights =  os.path.join(os.getcwd(), 'DataFiles/transfer_learning.h5')
