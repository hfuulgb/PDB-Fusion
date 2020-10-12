import numpy as np
import matplotlib.pyplot as plt

# 用来正常显示负号
plt.rcParams["axes.unicode_minus"] = False

# 刻度大小
plt.rcParams["axes.labelsize"] = 12
# 线的粗细
plt.rcParams["lines.linewidth"] = 1
# x轴标签大小
plt.rcParams["xtick.labelsize"] = 12
# y轴标签大小
plt.rcParams["ytick.labelsize"] = 12
# 图例大小
plt.rcParams["legend.fontsize"] = 12
# 图大小
plt.rcParams["figure.figsize"] = [8, 5]

# 生成-5~5 的10个数 array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4])
x = [500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
y = [
    0.5929,
    0.6019,
    0.6135,
    0.6091,
    0.6082,
    0.6013,
    0.6119,
    0.6081,
    0.6051,
    0.5946,
    0.5867,
]

# 正常显示中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
# plt.rcParams['font.sans-serif'] = ['SimHei']

# max_indx=np.argmax(y)#max value index
# min_indx=np.argmin(y)#min value index

# 绘图，设置(label)图例名字为'第一条线'，显示图例plt.legend()
plt.plot(x, y, label="MCC", marker="o", mfc="orange", ms=12, alpha=0.7, mec="c")

# x轴标签
plt.xlabel("最大截断序列长度 (L)")
# y轴标签
plt.ylabel("马修斯相关系数MCC")

# 可视化图标题
plt.title("(a)")

# 显示图例
plt.legend()

max_indx = np.argmax(y)  # max value index
min_indx = np.argmin(y)  # min value index
plt.plot(x[max_indx], y[max_indx], "rs")
show_max = "[" + str(x[max_indx]) + ", " + str(y[max_indx]) + "]"
plt.annotate(show_max, xytext=(x[max_indx], y[max_indx]), xy=(x[max_indx], y[max_indx]))
plt.plot(x[min_indx], y[min_indx], "gs")

# 显示图形
plt.show()
