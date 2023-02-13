I have implemented word2vec skip-gram embedding model using Negative Sampling. 

### Dataset

check [this](https://github.com/raja-kumar/CSE-244-ML-for-NLP/tree/main/asg1#dataset)

### Details

#### creating the vocab and training data
To create the vocab, I have used a threshhold of 50 word count. For training data preparation, I have used a context window of size of 2 and for each context word I used four negative samples.

#### Objective function
![alt text](https://github.com/raja-kumar/CSE-244-ML-for-NLP/blob/main/asg1/word2vec/imgs/w2v_obj_fn.png)

Refer to the [original paper](https://arxiv.org/pdf/1301.3781.pdf) for more details

### Training the model and creating the word embedding

if you want to do training, uncommnent the training line in run.sh.

```
./run.sh

```
