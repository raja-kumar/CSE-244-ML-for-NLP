import os
import numpy as np
from matplotlib import pyplot as plt
import re
from utils import *
import random

RE_ART = re.compile(r'\b(a|an|the)\b')
RE_PUNC = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def remove_articles(_text):
    return RE_ART.sub(' ', _text)


def white_space_fix(_text):
    return ' '.join(_text.split())


def remove_punc(_text):
    return RE_PUNC.sub(' ', _text)  # convert punctuation to spaces


def lower(_text):
    return _text.lower()


def normalize(text):
    """Lower text and remove punctuation, articles and extra whitespace. """
    return white_space_fix(remove_articles(remove_punc(lower(text))))


class createDataset():

    def __init__(self, path) -> None:
        with open(path, 'r') as f:
            self.data = f.read(30000000)
        self.vocab = {}
        self.index_to_word ={}

    def create_vocab_v2(self):

        with open('./../vocab.txt', 'r') as f:
            vocab_data = f.read().split('\n')
        
        new_vocab = {}
        new_index_to_word = {}

        i = 1
        new_vocab['ukn'] = 0
        new_index_to_word[0] = 'ukn'

        for word in vocab_data:
            new_vocab[word] = i
            new_index_to_word[i] = word
            i += 1
        
        return new_vocab, new_index_to_word
        
    
    def clean_data(self):
        #articles = ['a', 'an', 'the', 'and', 'or']
        self.data = normalize(self.data).split(' ')
    
    def create_vocab(self):
        self.word_count = {}
        #self.index_to_word = {}
        
        i = 1
        self.vocab['ukn'] = 0
        self.index_to_word[0] = 'ukn'
        for word in self.data:
            if word in self.word_count:
                self.word_count[word] += 1
                if (self.word_count[word] == 50): #minimum word frequency
                    self.vocab[word] = i
                    self.index_to_word[i] = word
                    i += 1
            else:
                self.word_count[word] = 1

        return self.vocab, self.index_to_word, self.word_count
    
    def one_hot_vector(self, word):
        self.vocab_size = len(self.vocab)
        oh = np.zeros(self.vocab_size)
        if not word in self.vocab:
            word = 'ukn'
        index = self.vocab[word]
        oh[index] = 1.0

        return oh
    
    def prepare_training_data(self):

        self.clean_data()
        print('data cleaned')
        #vocab, index_to_word, wc = self.create_vocab()
        vocab, index_to_word = self.create_vocab_v2()
        print('index created')
        corpus = self.data
        n = len(corpus)
        X_train = []
        Y_train = []
        count_th = 10000
        #print(n)
        print(len(corpus))
        print('vocab size', len(vocab))

        print("training data preparation started")

        for i in range(1,len(corpus)-1):
            #or wc[corpus[i]] > count_th
            #and self.word_count[corpus[i]] > count_th

            if(not corpus[i] in vocab ):
                continue
            
            if (corpus[i+1] in vocab):
                X_train.append(str(corpus[i]))
                curr_y = []
                curr_y.append(corpus[i+1]) #add positive sample
                negative_indices = random.sample(range(0, len(vocab)), 2) # pick 4 random negative samples
                for n_i in negative_indices:
                    curr_y.append(index_to_word[n_i])
                Y_train.append(curr_y)
            
            if (corpus[i-1] in vocab):
                X_train.append(str(corpus[i]))
                curr_y = []
                curr_y.append(corpus[i-1]) #add positive sample
                negative_indices = random.sample(range(0, len(vocab)-1), 2) # pick 4 random negative samples
                #print(negative_indices)
                for n_i in negative_indices:
                    curr_y.append(index_to_word[n_i])
                Y_train.append(curr_y)
                
        print("training data preparation finished")
        
        return np.array(X_train), np.array(Y_train), vocab, index_to_word
    
    def prepare_test_set(self, vocab_path):
        with open(vocab_path, 'r') as f:
            test_words = f.read().split('\n')
        
        #print(test_words)
        X_test = []

        for word in test_words:
            curr_oh = self.one_hot_vector(word)
            X_test.append(curr_oh)
        
        return np.array(X_test)