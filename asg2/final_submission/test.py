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
from train import compute_metrics


def test():

    ckpt = "./checkpoints/test_trainer/checkpoint-4000"

    test_dataset = prepareTestData()
    train_dataloader = prepareTrainingData()
    eval_dataloader = prepareValData()


    training_args = TrainingArguments(output_dir="./pred_ckpts", evaluation_strategy="epoch", num_train_epochs=10)
    trained_model = AutoModelForSequenceClassification.from_pretrained(ckpt, local_files_only=True)

    
    trainer = Trainer(
        model=trained_model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=eval_dataloader,
        compute_metrics=compute_metrics,
    )

    trainer.model = trained_model.cuda()
    #trainer.evaluate()

    test_labels = trainer.predict(test_dataset=test_dataset)
    test_labels = np.argmax(test_labels[0], axis=-1)

    mapping = {0:'barely-true', 1:'false', 2:'half-true', 3:'mostly-true', 4:'pants-fire',
       5:'true'}

    predic_file = './predictions.txt'

    with open(predic_file, 'w') as f:
        for j in test_labels:
            f.write(mapping[j])
            f.write('\n')
    print("result file generated at ", predic_file)    