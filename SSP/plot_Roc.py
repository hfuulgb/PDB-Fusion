import numpy as np
import matplotlib.pyplot as plt

# 用来正常显示负号
plt.rcParams['axes.unicode_minus']=False

# 刻度大小
plt.rcParams['axes.labelsize']=12
# 线的粗细
plt.rcParams['lines.linewidth']=1
# x轴标签大小
plt.rcParams['xtick.labelsize']=12
# y轴标签大小
plt.rcParams['ytick.labelsize']=12
#图例大小
plt.rcParams['legend.fontsize']=12
# 图大小
plt.rcParams['figure.figsize']=[8,5]

# 生成-5~5 的10个数 array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4])
x = [500,550,600,650,700,750,800,850,900,950,1000]
y = [0.8795,0.8829,0.8883,0.8857,0.8867,0.8843,0.8868,0.8854,0.8821,0.8790,0.8770] 

# 正常显示中文字体
# plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['font.sans-serif'] = ['SimHei'] 

# 绘图，设置(label)图例名字为'第一条线'，显示图例plt.legend()
plt.plot(x,y,label='AUC',marker='p',mfc='orange',ms=12,alpha=0.7,mec='c') 

# x轴标签
plt.xlabel('最大序列截断长度 (L)')
# y轴标签
plt.ylabel('ROC曲线下方的面积AUC')

# 可视化图标题
plt.title('(b)')

# 显示图例
plt.legend()

max_indx=np.argmax(y)#max value index
min_indx=np.argmin(y)#min value index
plt.plot(x[max_indx],y[max_indx],'rs')
show_max='['+str(x[max_indx])+', '+str(y[max_indx])+']'
plt.annotate(show_max,xytext=(x[max_indx],y[max_indx]),xy=(x[max_indx],y[max_indx]))
plt.plot(x[min_indx],y[min_indx],'gs')
# 显示图形
plt.show() 