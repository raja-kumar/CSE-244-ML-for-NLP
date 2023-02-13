import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import re
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
        self.index_to_text = {}
    
    def clean_data(self):
        #articles = ['a', 'an', 'the', 'and', 'or']
        self.data = normalize(self.data).split(' ')
    
    def create_vocab(self):
        self.word_count = {}
        
        i = 1
        self.vocab['ukn'] = 0
        self.index_to_text[0] = 'ukn'
        for word in self.data:
            if word in self.word_count:
                self.word_count[word] += 1
                if (self.word_count[word] == 50): #minimum word frequency
                    self.vocab[word] = i
                    self.index_to_text[i] = word
                    i += 1
            else:
                self.word_count[word] = 1

        return self.vocab, self.index_to_text
    
    def createCoOccureneceMat(self, window_size):

        vocab, index_to_word = self.create_vocab()
        vocab_size = len(vocab)
        self.coOcMat = np.zeros((vocab_size, vocab_size))
        corpus = self.data
        n = len(self.data)
        for i in range(window_size, n-window_size):
            curr_word = corpus[i]
            #
            if (curr_word in vocab and self.word_count[curr_word] < 10000):
                word_index = vocab[curr_word]
                for j in range(i+1, i+window_size):
                    if (corpus[j] in vocab):
                        curr_neigh_index = vocab[corpus[j]]
                        x = min(word_index, curr_neigh_index)
                        y = max(word_index, curr_neigh_index)
                        self.coOcMat[x][y] += 1

        return self.coOcMat, vocab, index_to_word