import os

base_model = True
seed=42

dense1 = 1024
dense2 = 512
dense3 = 128
keep_prob = 0.3
n_features = 48

train_classes = 63

learning_rate = 0.0001
batch_size = 128
num_epoches = 100
validation_split = 0.1

custom_acc = 0.97

#data paths and model weights
scalar_weights = os.path.join(os.getcwd(), 'Weights/scalar.pickle')
encoder_weights = os.path.join(os.getcwd(), 'Weights/encoder.pickle')
model_weights =  os.path.join(os.getcwd(), 'Weights/model.h5')

# train_csv = os.path.join(os.getcwd(), 'Datasets 0.5s/Train0.5s.csv')
# test_csv = os.path.join(os.getcwd(), 'Datasets 0.5s/Test0.5s.csv')
train_csv = os.path.join(os.getcwd(), '0.2 csv/Train0.2s.csv')
test_csv = os.path.join(os.getcwd(), '0.2 csv/Test0.2s.csv')