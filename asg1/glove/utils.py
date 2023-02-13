import numpy as np
import pickle

def softmax(arr):
    return np.exp(arr)/sum(np.exp(arr))

def error(pred,y):
    return - np.sum(np.log(pred)*y)

def load_model(model_path):
    model_file = open(model_path, 'rb')
    weights = pickle.load(model_file)
    model_file.close()

    return weights

def one_hot_vector(word, vocab):
    vocab_size = len(vocab)
    oh = np.zeros(vocab_size)
    if not word in vocab:
        word = 'ukn'
    index = vocab[word]
    oh[index] = 1.0

    return oh

def save_model(file_name, model_weights):
    with open(file_name, 'wb') as f:
        pickle.dump(model_weights, f)
    