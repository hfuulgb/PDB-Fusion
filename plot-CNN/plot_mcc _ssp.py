#!/usr/bin/env python
# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

plt.rcParams[u"font.sans-serif"] = ["simhei"]
plt.rcParams["axes.unicode_minus"] = False

# grid = plt.GridSpec(nrows=2, ncols=3, wspace=0.2, hspace=0.2)
# plt.figure(figsize = (12, 6))
# plt.rcParams['axes.unicode_minus']=False

# 刻度大小
plt.rcParams["axes.labelsize"] = 10
# 线的粗细
plt.rcParams["lines.linewidth"] = 1
# x轴标签大小
plt.rcParams["xtick.labelsize"] = 10
# y轴标签大小
plt.rcParams["ytick.labelsize"] = 10
# 图例大小
plt.rcParams["legend.fontsize"] = 10
# 图大小
plt.rcParams["figure.figsize"] = [5, 4]


n_groups = 4

means_k1 = (61.72, 64.58, 63.91, 61.45)
std_k1 = (1.39, 1.60, 1.25, 3.67)

means_k2 = (62.37, 63.91, 63.59, 62.43)
std_k2 = (1.32, 1.52, 1.82, 2.67)


fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.2

opacity = 0.4
error_config = {"ecolor": "0.3"}

rects1 = ax.bar(
    index,
    means_k1,
    bar_width,
    alpha=opacity,
    color="b",
    yerr=std_k1,
    error_kw=error_config,
    label="DBP-CNN(Length=600)",
)

rects2 = ax.bar(
    index + bar_width,
    means_k2,
    bar_width,
    alpha=opacity,
    color="r",
    yerr=std_k2,
    error_kw=error_config,
    label="DBP-CNN(Length=800)",
)


for a, b in zip(index, means_k1):
    plt.text(a, b, "%.2f" % b, ha="center", va="bottom", fontsize=9)

for a, b in zip(index, means_k2):
    plt.text(a + 0.2, b, "%.2f" % b, ha="center", va="bottom", fontsize=9)


ax.set_xlabel(u"CNN layes")
ax.set_ylabel(u"MCC(%)")
ax.set_title(u"(a)")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(("L=2", "L=3", "L=4", "L=5"))
ax.legend()
plt.ylim((60, 70))
fig.tight_layout()

plt.show()
