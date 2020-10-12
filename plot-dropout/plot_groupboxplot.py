import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

filename="plot-dropout/dropout_ex.csv"
# filename="dropout_ex.csv"
 
trainData =pd.read_csv(filename)#导入csv文件
 # plt.show()

# sns.set(style="whitegrid")
# plt.figure(figsize = [12, 5])
# base_color = sns.color_palette()[2]
# left plot: violin plot
# plt.subplot(1, 2, 1)
# plt.xlim((0,1))  #x轴刻度范围
# plt.ylim((0.5,0.7))  #y轴刻度范围
plt.title("(a) MCC Performence ", loc="center")

# ax1 = sns.boxplot(data = df, x = 'dropout',y = 'MCC',hue="model", width=0.7,palette="Set1")

testPlot = sns.boxplot(x='model', y='MCC', hue='dropout', data=trainData)
m1 = trainData.groupby(['model', 'dropout'])['MCC'].median().values
mL1 = [str(np.round(s, 1)) for s in m1]

ind = 0
for tick in range(len(testPlot.get_xticklabels())):
    testPlot.text(tick, m1[ind+1]+1, mL1[ind+1],  horizontalalignment='center',  color='b', weight='semibold')
    # testPlot.text(tick+.2, m1[ind]+1, mL1[ind], horizontalalignment='center', color='w', weight='semibold')
    ind += 2    
plt.show()