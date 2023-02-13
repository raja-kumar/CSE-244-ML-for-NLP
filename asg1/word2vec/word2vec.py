import os
import numpy as np
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm

from utils import *

class model():

    def __init__(self, X_train, Y_train, Vocab, embedding_size) -> None:
        # self.vocab_size = 
        self.x_train = X_train
        self.y_train = Y_train
        self.embedding_shape = embedding_size
        self.Vocab = Vocab
        self.vocab_size = len(self.Vocab)
        self.w_hidden = np.random.rand(self.vocab_size, self.embedding_shape)
        self.w_out = np.random.randn(self.embedding_shape, self.vocab_size)
        self.weights = {'w_hidden': self.w_hidden, 'w_out':self.w_out}

        # self.weights = load_model('./model_20_5epoch.pickle')
        # self.w_hidden = self.weights['w_hidden']
        # self.w_out = self.weights['w_out']
    
    def forward(self, input, train=True):
        o_hidden = np.matmul(input, self.w_hidden)
    
    def sigmoid(self, x):
        #print(x)
        return 1/(1+np.exp(-x))
    
    def encode(self, word):
        ## word is one-hot encoded

        word = np.reshape(word, (1,-1))
        encoding = np.matmul(word, self.w_hidden)
        return encoding
    
    def backprop(self, X, e_input, y, lr, cache):

        ## update the out layer weights
        context_index = self.Vocab[y[0]]
        d_out_context = (1-cache['context_word'])*e_input
        #d_out_context = -1*(1-cache['context_word'])*e_input

        neg_word_1_index = self.Vocab[y[1]]
        d_out_neg_1 = -1*(1-cache['negative_1'])*e_input
        #d_out_neg_1 = (1-cache['negative_1'])*e_input
        

        neg_word_2_index = self.Vocab[y[2]]
        d_out_neg_2 = -1*(1-cache['negative_2'])*e_input
        #d_out_neg_2 = (1-cache['negative_2'])*e_input

        ## update the embedding layer weights
        context_word = one_hot_vector(y[0], self.Vocab)
        neg_word_1 = one_hot_vector(y[1], self.Vocab)
        neg_word_2 = one_hot_vector(y[2], self.Vocab)
        theta_context = np.reshape(np.matmul(self.w_out, context_word), (1,-1))
        theta_negative_1 =  np.reshape(np.matmul(self.w_out, neg_word_1),(1,-1))
        theta_negative_2 = np.reshape(np.matmul(self.w_out, neg_word_2), (1,-1))
        X = np.reshape(X, (1,-1))
        d_hidden = np.matmul(X.T, theta_context)*(1-cache['context_word']) + np.matmul(X.T, theta_negative_1)*(1-cache['negative_1']) + np.matmul(X.T, theta_negative_2)*(1-cache['negative_2'])
        #d_hidden = theta_context
        self.w_out[:,context_index] += lr*d_out_context[0]
        self.w_out[:, neg_word_1_index] += lr*d_out_neg_1[0]
        self.w_out[:, neg_word_2_index] += lr*d_out_neg_2[0]

        self.w_hidden += lr*d_hidden

    
    def error_negative_sampling(self, e_input, y):
        cache = {}

        context_word = one_hot_vector(y[0], self.Vocab)
        theta_context = np.matmul(self.w_out, context_word)
        temp_context_word = self.sigmoid(np.dot(e_input, theta_context))
        curr_loss = np.log(temp_context_word)
        #curr_loss = -1*np.log(temp_context_word)
        cache['context_word'] = temp_context_word

        negative_word_1 = one_hot_vector(y[1], self.Vocab)
        theta_negative_1 = np.matmul(self.w_out, negative_word_1)
        temp_neg_1 = self.sigmoid(-1*np.dot(e_input, theta_negative_1))
        #temp_neg_1 = self.sigmoid(np.dot(e_input, theta_negative_1))
        curr_loss += np.log(temp_neg_1)
        cache['negative_1'] = temp_neg_1

        negative_word_2 = one_hot_vector(y[2], self.Vocab)
        theta_negative_2 = np.matmul(self.w_out, negative_word_2)
        temp_neg_2 = self.sigmoid(-1*np.dot(e_input, theta_negative_2))
        #temp_neg_2 = self.sigmoid(np.dot(e_input, theta_negative_2))
        curr_loss += np.log(temp_neg_2)
        cache['negative_2'] = temp_neg_2
        
        return -curr_loss, cache
    
    def train(self):

        lr = 0.01
        num_epoch = 5

        self.error_list = []

        for i in range(num_epoch):

            # if(i%20 == 0):
            #     lr = lr/10

            curr_error = []
            total = len(self.x_train)
            for input_, y in tqdm(zip(self.x_train, self.y_train), total=total):

                input_oh = one_hot_vector(input_, self.Vocab)
                e_input = self.encode(input_oh)
                error_, cache = self.error_negative_sampling(e_input, y)
                curr_error.append(error_)

                self.backprop(input_oh,e_input,y,lr,cache)

            self.error_list.append(np.average(curr_error))
            print("epoch:", i, "avg_error", np.average(curr_error))
        
        self.weights['w_hidden'] = self.w_hidden
        self.weights['w_out'] = self.w_out

        return self.weights
    
    def create_embedding_file(self, vocab_path, X_test):
        with open(vocab_path, 'r') as f:
            test_words = f.read(1000).split('\n')
        
        output_file = open('./vectors.txt', 'w')

        for word, oh_word in zip(test_words, X_test):
            curr_embedding = self.encode(oh_word)
            emb_string = ''
            for v in curr_embedding[0]:
                emb_string += str(v) + ' '
            output_file.write(word + ' ' + emb_string[:-1] + '\n')
        
        output_file.close()

    
    def plot_loss(self):
        plt.plot(self.error_list)
