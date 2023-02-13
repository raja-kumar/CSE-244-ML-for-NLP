from utils import *
from dataset import createDataset

def create_embedding_file(vocab_path, Vocab, embedding, output_file):
        with open(vocab_path, 'r') as f:
            test_words = f.read().split('\n')
        
        output_file = open(output_file, 'w')

        for word in test_words:
            oh_word = one_hot_vector(word, Vocab)
            curr_embedding = np.matmul(oh_word, embedding)
            emb_string = ''
            #print(curr_embedding)
            for v in curr_embedding:
                v = np.round(v,3)
                #print(v)
                emb_string += str(v) + ' '
            output_file.write(word + ' ' + emb_string[:-1] + '\n')
        
        output_file.close()
        
if __name__ == "__main__":
    ds = createDataset('./../text8')
    X_train,Y_train, Vocab, index_to_word = ds.prepare_training_data()
    print(' training data loaded')

    model = load_model('./word2vec20_epoch10.pickle')

    create_embedding_file('./../vocab.txt', Vocab, model['w_hidden'], './vectors.txt')

    print("word embedding save in vectors.txt")