# encoding:utf-8

import copy
import codecs
import pickle
import random
import numpy as np
from pathlib import Path

cur_path = Path(__file__).absolute().parent
with codecs.open(cur_path / 'data/embedding.pt', 'rb') as file:
    EMBEDDING = pickle.load(file)

MAX_LEN = 20

class Data(object):
    def __init__(self, data_path, label_path):
        self.train_X, self.train_y = self._read(data_path), self._read(label_path)
        assert len(self.train_X) == len(self.train_y)

    def _read(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            lines = file.read().split('\n')[:-1]
        return lines
    
    def get(self, batch_size):
        """do not use test data here"""
        self._shuffle()
        data_size = len(self.train_X)
        if data_size % batch_size == 0:
            batch_number = int(data_size / batch_size)
        else:
            batch_number = int(data_size // batch_size + 1)
        for bn in range(batch_number):
            if bn < batch_number - 1:
                start = bn * batch_size
                end = start + batch_size
            else:
                start = bn * batch_size
                end = None
            train_X_batch = self.train_X[start : end]
            train_y_batch = self.train_y[start : end]

            train_X_processed = np.expand_dims(np.array(list(map(self._convert_to_embedding, train_X_batch))), axis=1)
            train_y_processed = np.array(list(map(self._convert_to_int, train_y_batch)))

            yield(train_X_processed, train_y_processed)

    def _convert_to_embedding(self, sentence):
        if len(sentence) < MAX_LEN:
            padding_str = ''.join(['*' for _ in range(MAX_LEN - len(sentence))])
            sentence += padding_str
        else:
            sentence = sentence[:MAX_LEN]
        assert len(sentence) == MAX_LEN    

        array = []
        for vocab in sentence:
            if vocab != '*' and vocab in EMBEDDING:
                array.append(EMBEDDING[vocab])
            elif vocab == '*':
                array.append(EMBEDDING['<pad>'])
            else:
                array.append(EMBEDDING['<unk>'])
        
        return np.array(array)
    
    def _convert_to_int(self, label):
        assert type(label) == str
        assert len(label) <= 2
        return int(label)
    
    def _shuffle(self):
        data_label = list(zip(self.train_X, self.train_y))
        random.shuffle(data_label)
        train_X_temp, train_y_temp = zip(*data_label)
        self.train_X = copy.deepcopy(list(train_X_temp))
        self.train_y = copy.deepcopy(list(train_y_temp))
    
    def __len__(self):
        return len(self.train_X)
    
if __name__ == '__main__':
    data = Data('train_x', 'train_y')

    for (X, y) in data.get(3):
        print(X[0:2, :].shape)
        print(type(y))
        input()