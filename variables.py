import os

base_model = True
seed=42
dense0 = 1024
dense1 = 512
dense2 = 512
dense3 = 128
dense4 = 128
denset = 32
keep_prob = 0.7
output_dim = 256
n_features = 48
batch_size = 128
num_epoches = 100
validation_split = 0.2

#data paths and model weights
pca_weights = os.path.join(os.getcwd(), 'Weights/pca.pickle')
confustion_img = os.path.join(os.getcwd(), 'NewData/Confusion Matrix.png')
train_csv = os.path.join(os.getcwd(), 'NewData/TrainData.csv')
test_csv  = os.path.join(os.getcwd(), 'NewData/TestData.csv')
model_path =  os.path.join(os.getcwd(), 'Weights/model.json')
model_weights =  os.path.join(os.getcwd(), 'Weights/model.h5')