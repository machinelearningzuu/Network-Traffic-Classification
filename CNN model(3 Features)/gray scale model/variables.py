import os
seed=42

dense1 = 1024
dense2 = 512
dense3 = 256
keep_prob = 0.5
n_features = 48
n_classes = 92
input_shape = (28,28,1)
input_size = 784

learning_rate = 0.001
batch_size = 128
num_epoches = 10
validation_split = 0.1
custom_acc = 0.9

#data paths
test_csv = 'Dataset/Test.csv'
train_csv = 'Dataset/Train.csv'
train_image_path = 'Dataset/images/'

#model weights
model_weights =  'Weights/model.h5'
scalar_weights = 'Weights/scalar.pickle'
encoder_weights = 'Weights/encoder.pickle'

#visualization
n_bins = 10
colors = ['#E69F00', '#56B4E9']
names = ['Train Confidence', 'Test Confidence']
confidence_img = 'visualization/confidence_distribution.png'

#Create main CSV
image_per_class = 2
inout_columns = ['frame.time_delta_displayed', 'frame.len', 'data.len']
vis_inout_columns = ['activity', 'in-frame.time_delta_displayed', 'in-frame.len', 'in-data.len', 'out-frame.time_delta_displayed', 'out-frame.len', 'out-data.len']