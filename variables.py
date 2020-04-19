import os

seed=42
Ntrain = 1100
n_features = 2
kernal_size = 3
pool_size = 3
output_dim = 1024
num_classes = 22
batch_size = 128
num_epoches = 30
validation_split = 0.2
frame_count_threshold = 100

hidden_dim = 256
dense1 = 64
dense2 = 64
dense3 = 64

#data paths and model weights
main_dir = os.path.join(os.getcwd(), 'FB')
model_path =  os.path.join(os.getcwd(), 'network_classifier.json')
model_weights =  os.path.join(os.getcwd(), 'network_classifier.h5')