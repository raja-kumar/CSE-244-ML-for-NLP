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

from dataset import (
    prepareTrainingData, 
    prepareValData,
    prepareTestData
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)

def train():
    model_name = "bert-base-cased"
    exp_name = "train_v1"
    num_epochs = 10

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
    training_args = TrainingArguments(output_dir=exp_name, evaluation_strategy="epoch", num_train_epochs=num_epochs)
    metric = evaluate.load("accuracy")

    train_dataloader = prepareTrainingData()
    eval_dataloader = prepareValData()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=eval_dataloader,
        compute_metrics=compute_metrics,
    )

    trainer.train()

