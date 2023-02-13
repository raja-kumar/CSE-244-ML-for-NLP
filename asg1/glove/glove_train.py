from dataset_glove import createDataset
from glove import model
from utils import *


if __name__ == "__main__":
    ds = createDataset('./../text8')
    ds.clean_data()
    co_occurance_mat, Vocab, index_to_word = ds.createCoOccureneceMat(10)
    print("------ cooccurance matrix created -------")

    embedding_size = 20
    lr = 1
    regularization_parameter = 0.5
    num_epoch = 5

    w_e = model(co_occurance_mat,embedding_size)
    #w_e = glove(co_occurance_mat, embedding_size, lr, regularization_parameter, num_epoch)

    print("-------- training completed ---------")
    #file_name = 'glove' + str(embedding_size) + "_" + str(num_epoch) + '.pickle'
    model_path = 'test.pickle'
    save_model(model_path, w_e)

    print("---------- model saved ---------")