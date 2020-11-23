import os

seed=42

kernal_size = 1
dense1 = 200
dense2 = 400
dense3 = 100
keep_prob = 0.1

n_features = 48

dim1 = 32
dim2 = 16
dim3 = 6

train_classes = 63

learning_rate = 0.0001
batch_size = 64
num_epoches = 10
validation_split = 0.15

custom_acc = 0.9

#data paths and model weights
scalar_weights = os.path.join(os.getcwd(), 'Weights/scalar.pickle')
encoder_weights = os.path.join(os.getcwd(), 'Weights/encoder.pickle')
model_weights =  os.path.join(os.getcwd(), 'SAE_CNN_weights/cnn_model.h5')
autoencoder_weights =  os.path.join(os.getcwd(), 'SAE_CNN_weights/autoencoder_model.h5')
sae_weights =  os.path.join(os.getcwd(), 'SAE_CNN_weights/sae_weights.h5')
final_model_weights = os.path.join(os.getcwd(), 'SAE_CNN_weights/final_model_weights.h5')

train_csv = os.path.join(os.getcwd(), 'Datasets 0.5s/Train0.5s.csv')
test_csv = os.path.join(os.getcwd(), 'Datasets 0.5s/Test0.5s.csv')
# train_csv = os.path.join(os.getcwd(), '0.2 csv/Train0.2s.csv')
# test_csv = os.path.join(os.getcwd(), '0.2 csv/Test0.2s.csv')

#Data Visulization
plot_step = 100
acc_img = "Visualization/accuracy_comparison.png"
loss_img = "Visualization/loss_comparison.png"

TrainApps = ['Fb', 'Gm', 'Msg', 'Ut', 'Vb', 'Sk']
TestApps = ['In', 'Wt']
img_corr = "Visualization/Correlation/{}_VS_{}.png"
FullAppNames = {
                'Fb' : 'FaceBook',
                'Gm' : 'Gmail',
                'Msg': 'Massenger',
                'Ut' : 'YouTube',
                'Vb' : 'Viber',
                'In' : 'Instagram',
                'Wt' : 'WhatsApp',
                'Sk' : 'Skype'
                }