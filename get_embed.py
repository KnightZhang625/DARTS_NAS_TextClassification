# encoding:utf-8

import codecs
import pickle
import numpy as np
from pathlib import Path

# set the pre_trained embedding path
cur_path = Path(__file__).absolute().parent
embedding_path = cur_path / 'data/chn_vecs'

# read pre_trained embedding file
with codecs.open(embedding_path, 'r', 'utf-8') as file:
    lines = file.read().split('\n')[1:10]     # cause the first line saves the meta information

# split vocab and vector -> transfer str to float -> convert to ndarray
voc_vec_split = lambda line : (line.split(' ')[0], np.array(list(map(float, line.split(' ')[1:301]))))
embedding = {voc_vec[0] : voc_vec[1] for voc_vec in list(map(voc_vec_split, lines))}

# add <unk>, add <padding>
np.random.seed(10)
embedding['<unk>'] = np.random.randn(300)
embedding['<pad>'] = np.random.randn(300)

# save the file
with codecs.open(cur_path / 'data/embedding.pt', 'wb') as file:
    pickle.dump(embedding, file)