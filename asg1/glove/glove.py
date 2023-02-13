import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import re
from utils import *

def model(co_occurance_mat, embedding_size):
    # initialize word embeddings with small random values
    vocab_size = len(co_occurance_mat)
    bias = 0.5
    num_epoch = 10
    lr = 1
    embedding_mat = 0.1*np.random.randn(vocab_size, embedding_size)
    co_sum = co_occurance_mat.sum()
    for epoch in range(num_epoch):
        error = 0

        for i, curr_word in tqdm(enumerate(co_occurance_mat), total=vocab_size):
            for j, curr_sim in enumerate(curr_word):
                if curr_sim == 0:
                    continue
                u_v = np.dot(embedding_mat[i], embedding_mat[j])
                Xij = (curr_sim / co_sum)** 0.75

                ## calculate the error
                error += Xij * (u_v + bias - np.log(curr_sim)) ** 2
                d_e = u_v + bias - np.log(curr_sim)
                
                ### backprop

                embedding_mat[i] -= lr * Xij * d_e * embedding_mat[j]
                embedding_mat[j] -= lr * Xij * d_e * embedding_mat[i]
        
        print('\n epoch ', epoch, error)
    return embedding_mat