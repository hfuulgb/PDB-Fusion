# -*- coding: utf-8 -*-
import numpy as np
import pdb


def read_seq(seq_file, mod="extend"):
    seq_list = []
    seq = ""
    with open(seq_file, "r") as fp:
        for line in fp:
            seq = line[:-1]
            seq_array = get_seq_concolutional_array(seq)
            seq_list.append(seq_array)

    return np.array(seq_list)


def get_seq_concolutional_array(seq):
    # seq = seq.replace('U', 'T')
    # except BJOUXZ
    alpha = "ACDEFGHIKLMNPQRSTVWY"
    row = len(seq)
    new_array = np.zeros((row, 20))

    for i, val in enumerate(seq):

        if val not in "ACDEFGHIKLMNPQRSTVWY":
            if val == "Z":
                new_array[i] = np.array([0.0] * 20)
            # if val == 'S':
            #     new_array[i] = np.array([0, 0.5, 0.5, 0, 0])
            continue

        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except ValueError:
            pdb.set_trace()
    return new_array


# ------------------------------------主函数---------------------------------------------

from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
from keras import backend as K
from keras.utils import np_utils
import numpy as np
import os


if __name__ == "__main__":
    # trueSet, falseSet = readfile('data/IE_true.seq', 'data/IE_false.seq', 0)
    seq_list = []
    seq = ""
    i = 0
    with open("data/DNA_Pading2_PDB14189", "r") as fp:
        for line in fp:
            seq = line[:-1]
            if len(seq) != 1000:
                print("[" + str(i) + "]:/-[" + str(len(seq)) + "]")
            i += 1

# from numpy import array
# from keras.preprocessing.text import one_hot
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers.embeddings import Embedding
# # define documents
# docs = ['Well done!',
# 		'Good work',
# 		'Great effort',
# 		'nice work',
# 		'Excellent!',
# 		'Weak',
# 		'Poor effort!',
# 		'not good',
# 		'poor work',
# 		'Could have done better.']
# # define class labels
# labels = array([1,1,1,1,1,0,0,0,0,0])
# # integer encode the documents
# vocab_size = 50
# encoded_docs = [one_hot(d, vocab_size) for d in docs]
# print(encoded_docs)
# # pad documents to a max length of 4 words
# max_length = 4
# padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs)
# # define the model
# model = Sequential()
# model.add(Embedding(vocab_size, 8, input_length=max_length))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
# # compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# # summarize the model
# print(model.summary())
# # fit the model
# model.fit(padded_docs, labels, epochs=50, verbose=0)
# # evaluate the model
# loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
# print('Accuracy: %f' % (accuracy*100))
