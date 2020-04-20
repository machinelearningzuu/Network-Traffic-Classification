import os

seed=42
n_features = 2
kernal_size = 3
pool_size = 2
output_dim = 256
num_classes = 9
batch_size = 32
num_epoches = 150
validation_split = 0.2
frame_count_threshold = 100

lstm1 = 128
lstm2 = 64
lstm3 = 32
dense1 = 64
dense2 = 64
dense3 = 64

#data paths and model weights
main_dir = os.path.join(os.getcwd(), 'FB_split')
model_path =  os.path.join(os.getcwd(), 'network_classifier.json')
model_weights =  os.path.join(os.getcwd(), 'network_classifier.h5')