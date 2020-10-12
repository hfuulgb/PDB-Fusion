# _*_ coding:utf-8 _*_
import numpy as np

dictionary={'Z':0,'B':0,'J':0,'O':0,'U':0,'X':0,'A':1,'C':2,
              'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,
              'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,
              'V':18,'W':19,'Y':20}

data_dir = './data/DNA_set_nopading_'
def trans_Sample(data_dir):
    with open(data_dir,'r+',encoding='utf-8') as file:
        context=file.readlines()
        context_samples=len(context)
        result_samples=[]
        for i in range(context_samples):
            context_current=context[i][:-1]
            result_record=[]
            for char in context_current:
                result_record.append(dictionary[char])
            result_samples.append(np.array(result_record))
        result_samples=np.array(result_samples)
        #print(result_samples.shape)
    return result_samples




# for i,j in zip(dictionaries.keys(),dictionaries.values()):
#     print(i,j)

if __name__=='__main__':

    trans_Sample(data_dir+'0')