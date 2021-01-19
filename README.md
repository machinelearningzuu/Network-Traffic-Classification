# Network Traffic Classification and Anomaly Detection

Deep Learning model for analysis and identify the application for given Teletraffic pattern. Experiment with different models including MLPs and CNNs. Final object 
is to detect anomaly apps with unusual traffic patterns.

  - Analyze network traffic for both incoming and outgoing
  - Extract statistical features
  - Train the supervised deep learning model 
  - Handle anomalies using softmax probabilities

# Techniques

  - Supervised Deep Learning, Unsupervised Deep Learning
  - Statistical feature calculation
 
# Tools

* TensorFlow - Deep Learning Model
* pandas - Data Extraction and Preprocessing
* numpy - numerical computations
* scikit learn - Advanced preprocessing

### Installation

Install the dependencies and conda environment

```sh
$ conda create -n envname python=python_version
$ activate envname 
$ conda install -c anaconda tensorflow-gpu
$ conda install -c anaconda pandas
$ conda install -c anaconda matplotlib
$ conda install -c anaconda scikit-learn
```

For Train Model...

```sh
$ python model.py
```
