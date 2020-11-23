import os
seed=42

dense1 = 1024
dense2 = 512
dense3 = 256
keep_prob = 0.5
n_features = 48
n_classes = 63

learning_rate = 0.0001
batch_size = 64
num_epoches = 20
validation_split = 0.15
custom_acc = 0.9

#data paths
test_csv = 'Data/Test.csv'
train_csv = 'Data/Train.csv'

#model weights
model_weights =  'Weights/model.h5'
scalar_weights = 'Weights/scalar.pickle'
encoder_weights = 'Weights/encoder.pickle'

#visualization
n_bins = 10
colors = ['#E69F00', '#56B4E9']
names = ['Train Confidence', 'Test Confidence']
confidence_img = 'visualization/confidence_distribution.png'