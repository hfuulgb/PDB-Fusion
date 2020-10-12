# -*- coding: utf-8 -*-
"""
@author: liguobin
"""

import random


def cutData(OrgSet, lens=14120, maxlen=600):
    cutedSet = []
    i = -1
    for line in OrgSet:
        i += 1
        if i % 2 != 1:
            continue
        # if i>lens:
        #     break
        temp = line.strip()
        lenth = len(temp)
        if lenth > maxlen:
            temp = temp[0:maxlen]
        if lenth <= maxlen:
            temp = temp.ljust(maxlen, "Z")
            # print(str(len(temp))+"          ")
        cutedSet.append(temp)
    return cutedSet


def readfile(trueSetFile, falseSetFile):
    ret = False
    try:
        OrgTrueSet = open(trueSetFile, "r")  # 打开文件
        OrgFalseSet = open(falseSetFile, "r")  # 打开文件
    except IOError:
        print("trueSetFile doesnot exist!")
        return ret

    # 逐行读取序列
    cutedTrueSet = cutData(OrgTrueSet)
    OrgTrueSet.close()

    # 逐行读取序列
    cutedfalseSet = cutData(OrgFalseSet, 14258)
    OrgFalseSet.close()

    # 数据比例选择
    return cutedTrueSet, cutedfalseSet


# ------------------------------------主函数---------------------------------------------

from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
from keras import backend as K
from keras.utils import np_utils
import numpy as np
import os

if __name__ == "__main__":
    path = "data/"
    if not os.path.exists(path):
        os.makedirs(path)

    trueSet, falseSet = readfile("DNA/PDB186_P.txt", "DNA/PDB186_N.txt")
    # t_train_list = []
    # tset = np.array(trueSet)

    print(len(trueSet))
    print(len(falseSet))
    # xglable = np.ones(1116)
    # yglable = np.zeros(1116)
    # xglable = np.ones(1116)
    # yglable = np.zeros(1116)
    # glabel = np.concatenate((xglable, yglable), axis=0)
    # # y_train =np.array(np.array(glabel))
    with open(path + "DNA_Pading_600_PDB186", mode="w", encoding="utf-8") as myfile:
        myfile.write("\n".join(trueSet))
        myfile.write("\n")
        myfile.write("\n".join(falseSet))
        myfile.write("\n")

    with open(path + "class_PDB186", mode="w", encoding="utf-8") as myfile:
        # for indexi in range(2796):
        for indexi in range(len(trueSet)):
            myfile.write("1" + "\n")
        # for indexi in range(2796):
        for indexi in range(len(falseSet)):
            myfile.write("0" + "\n")
