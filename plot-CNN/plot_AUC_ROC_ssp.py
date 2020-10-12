#!/usr/bin/env python
#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

# plt.rcParams[u'font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False

# grid = plt.GridSpec(nrows=2, ncols=3, wspace=0.2, hspace=0.2)
# plt.figure(figsize = (12, 6))
# plt.rcParams['axes.unicode_minus']=False

# 刻度大小
plt.rcParams['axes.labelsize']=10
# 线的粗细
plt.rcParams['lines.linewidth']=1
# x轴标签大小
plt.rcParams['xtick.labelsize']=10
# y轴标签大小
plt.rcParams['ytick.labelsize']=10
#图例大小
plt.rcParams['legend.fontsize']=10
# 图大小
plt.rcParams['figure.figsize']=[6,4]




n_groups = 3

means_k1 = (0.984, 0.986, 0.987)
std_k1 = (0.005, 0.004, 0.005)

means_k2 = (0.985, 0.987, 0.988)
std_k2 = (0.004, 0.004, 0.004)

means_k3 = (0.987, 0.986, 0.987)
std_k3 = (0.006, 0.004, 0.004)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.2

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, means_k1, bar_width,
                alpha=opacity, color='b',
                yerr=std_k1, error_kw=error_config,
                label='K-MER(K=1)')

rects2 = ax.bar(index+ bar_width , means_k2, bar_width,
                alpha=opacity, color='r',
                yerr=std_k2, error_kw=error_config,
                label='K-MER(K=2)')

rects3 = ax.bar(index + bar_width*2, means_k3, bar_width,
                alpha=opacity, color='g',
                yerr=std_k2, error_kw=error_config,
                label='K-MER(K=3)')

for a,b in zip(index,means_k1):
    plt.text(a, b, '%.3f' % b, ha='center', va= 'bottom',fontsize=7)

for a,b in zip(index,means_k2):
    plt.text(a+0.2, b, '%.3f' % b, ha='center', va= 'bottom',fontsize=7)


for a,b in zip(index,means_k3):
    plt.text(a+0.4, b, '%.3f' % b, ha='center', va= 'bottom',fontsize=7)

ax.set_xlabel(u'CNN层数L')
ax.set_ylabel(u'ROC曲线下面积(AUC)')
ax.set_title(u'( a )')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('L=1', 'L=2', 'L=3'))
ax.legend()
plt.ylim((0.98, 1.0))
fig.tight_layout()

plt.show()