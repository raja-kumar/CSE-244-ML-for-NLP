import os
import numpy as np
from matplotlib import pyplot as plt

from word2vec import model
from dataset import createDataset
from utils import *

if __name__ == "__main__":
    ds = createDataset('./../text8')
    X_train,Y_train, Vocab, index_to_word = ds.prepare_training_data()
    print(' training data loaded')

    embedding_size = 20
    num_epoch = 20
    
    model1 = model(X_train, Y_train, Vocab, embedding_size)
    model_weights = model1.train()
    
    print("-------- training completed ---------")
    file_name = 'word2vec' + str(embedding_size) + "_" + str(num_epoch) + '.pickle'
    #file_name = 'model_100_reduced_data.pickle'
    model_path = file_name
    save_model(model_path, model_weights)

    model1.plot_loss()