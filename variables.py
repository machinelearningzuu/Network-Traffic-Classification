import os

base_model = False
seed=42
dense1 = 512
dense2 = 512
dense3 = 128
dense4 = 128
denset = 64
keep_prob = 0.3
output_dim = 256
n_features = 48
batch_size = 128
num_epoches = 100
validation_split = 0.2

#data paths and model weights
big_csv = os.path.join(os.getcwd(), 'FGISY0.5s.csv')
tl_path = os.path.join(os.getcwd(), 'M0.5s.csv') # tl - transfer_learning
model_path =  os.path.join(os.getcwd(), 'base_model.json')
model_weights =  os.path.join(os.getcwd(), 'base_model.h5')
tl_model_path =  os.path.join(os.getcwd(), 'massenger.json')
tl_model_weights =  os.path.join(os.getcwd(), 'massenger.h5')