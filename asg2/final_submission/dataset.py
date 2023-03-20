import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
import csv

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        #self.train = train

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        #if (self.train):
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

class testDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        #self.labels = labels
        #self.train = train

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        #if (self.train):
        #item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

def preProcess(dataset="train"):
    if (dataset=="train"):
        df = pd.read_csv('./liar_dataset/train.tsv', sep='\t', header=None, quoting=csv.QUOTE_NONE) 
    elif(dataset=="val"):
        df = pd.read_csv('./liar_dataset/valid.tsv', sep='\t', header=None, quoting=csv.QUOTE_NONE)
    else:
        df = pd.read_csv('./liar_dataset/test.tsv', sep='\t', header=None, quoting=csv.QUOTE_NONE)

    data = {'labels': df[0].values, 'text' : df[1].values}
    labels = np.unique(data['labels'])
    pre = preprocessing.LabelEncoder()
    pre.fit(labels)
    train_labels = pre.transform(data['labels'])
    train_text = list(data['text'])

    return train_text, train_labels

def prepareTrainingData():

    train_text, train_labels = preProcess()

    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

    encoded_in = tokenizer(train_text,padding=True, truncation=True, return_tensors='pt')
    trainDataset = Dataset(encoded_in, train_labels)

    return trainDataset

def prepareValData():

    train_text, train_labels = preProcess("val")

    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    encoded_in = tokenizer(train_text,padding=True, truncation=True, return_tensors='pt')
    trainDataset = Dataset(encoded_in, train_labels)

    return trainDataset

def prepareTestData():

    df = pd.read_csv('./liar_dataset/test.tsv', sep='\t', header=None, quoting=csv.QUOTE_NONE)
    print("test data shape", df.shape)
    test_text = list(df[0].values)

    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_in = tokenizer(test_text,padding=True, truncation=True, return_tensors='pt')
    test_dataset = testDataset(encoded_in)

    return test_dataset