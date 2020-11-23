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
num_epoches = 200
validation_split = 0.15
custom_acc = 0.9

#data paths and model weights
scalar_weights = os.path.join(os.getcwd(), 'Weights/scalar.pickle')
encoder_weights = os.path.join(os.getcwd(), 'Weights/encoder.pickle')
model_weights =  os.path.join(os.getcwd(), 'Weights/model.h5')

train_csv = os.path.join(os.getcwd(), 'Data/Train.csv')
test_csv = os.path.join(os.getcwd(), 'Data/Test.csv')