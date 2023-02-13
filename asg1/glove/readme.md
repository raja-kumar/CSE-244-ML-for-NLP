I have implemented glove embedding model using Weighted Least Squares Regression Model.

### Dataset

check [this](https://github.com/raja-kumar/CSE-244-ML-for-NLP/tree/main/asg1#dataset)

### Details

#### creating the vocab and co-occurence matrix
To create the vocab, I have used a threshhold of 50 word count. For co-occurence matrix, I have used a window size of 10 to count the pairs.

#### Objective function
![alt text](https://github.com/raja-kumar/CSE-244-ML-for-NLP/blob/main/asg1/glove/imgs/glove_objective_fn.png)

Refer to the [original paper](https://nlp.stanford.edu/pubs/glove.pdf) for more details

### Training the model and creating the word embedding

if you want to do training, uncommnent the training line in run.sh.

```
./run.sh

```
