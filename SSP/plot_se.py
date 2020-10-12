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
y = [84.26, 85.30, 84.17, 84.39, 83.57, 82.78, 83.21, 82.73, 83.41, 83.27, 83.21]

# 正常显示中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
# mpl.rcParams['axes.unicode_minus'] = False

# 绘图，设置(label)图例名字为'第一条线'，显示图例plt.legend()
plt.plot(x, y, label="SE", marker="s", mfc="orange", ms=12, alpha=0.7, mec="c")

# x轴标签
plt.xlabel("最大序列截断长度 (L)")
# y轴标签
plt.ylabel("敏感度SE(%)")

# 可视化图标题
plt.title("(e)")

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
