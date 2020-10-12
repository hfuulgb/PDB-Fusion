# -*- coding: utf-8 -*-
"""
@author: liguobin
"""

import random


def readfile(trueSetFile):
    ret = False
    alpha = "ACDEFGHIKLMNPQRSTVWY"
    try:
        OrgtrueSet = open(trueSetFile, "r")  # 打开文件
    except IOError:
        print("trueSetFile doesnot exist!")
        return ret

    for seq in OrgtrueSet:
        # seq=seq[0:1000]
        for char in alpha:
            index = alpha.index(char) + 1
            seq = seq.replace(char, str(index) + " ")

        seq2 = seq[0 : len(seq) - 2]

        with open(
            "data/DNA_Encoding1_600_PDB1075", mode="a", encoding="utf-8"
        ) as myfile:
            myfile.write(str(seq2) + "\n")

    return


from keras import backend as K
from keras.utils import np_utils
import numpy as np
import os


if __name__ == "__main__":
    readfile("data/DNA_NoPading_600_PDB1075")
    print("finish!")
