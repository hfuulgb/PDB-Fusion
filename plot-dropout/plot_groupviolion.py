import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filename = "plot-dropout/dropout_ex.csv"
# filename="dropout_ex.csv"

df = pd.read_csv(filename)  # 导入csv文件

# 仅提取考试分数相关的信息
# d_scores = all_data[['ACC', 'MCC']]
# print(d_scores.head())
# d_scores_array = np.array(d_scores)
# fig,axes=plt.subplots(2,1)
#
# sns.boxplot(x="model", y="MCC", hue="drop", data=df,palette="Pastel1");

# plt.show()

# sns.boxplot(x="model", y="ACC", hue="drop", data=df,palette="Pastel1");

# plt.show()

sns.set(style="whitegrid")
plt.figure(figsize=[12, 5])
# base_color = sns.color_palette()[2]
# left plot: violin plot
plt.subplot(1, 2, 1)
# plt.xlim((0,1))  #x轴刻度范围
# plt.ylim((0.5,0.7))  #y轴刻度范围
plt.title("(a) MCC Performence ", loc="center")

ax1 = sns.violinplot(
    data=df, x="model", y="MCC", hue="dropout", width=0.7, palette="Set1"
)

# right plot: box plot
plt.subplot(1, 2, 2)
plt.title("(b) AUC Performence", loc="center")
ax2 = sns.violinplot(
    data=df, x="model", y="AUC", hue="dropout", width=0.7, palette="Set1"
)
# plt.ylim(ax1.get_ylim()) # set y-axis limits to be same as left plot

plt.show()
