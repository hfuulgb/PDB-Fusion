# _*_coding:UTF-8_*_
"""
对fastq文件中的序列进行处理
1.获取序列的id和序列信息
2.统计每个id对应的序列的长度
3.对序列长度进行统计
"""
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def read_fastq_seq(file_name):
    with open(file_name, "r") as f:
        raw_index = -1
        for line in f:
            raw_index += 1
            seq_id.append(raw_index)
            seq.append(line.rstrip())
    seq_dic = dict(zip(seq_id, seq))  # 将序列编号和序列信息两个列表合并成字典
    for v in seq_dic.values():  # 统计序列的长度
        seq_len.append(len(v))
    return seq_id, seq, seq_len, seq_dic


def count_reads_rate(seq, seq_len):
    # 统计总的reads数
    total_reads = len(seq)
    # print(total_reads)
    # 统计每个长度的reads数量，并计算其比值
    # 新建一个字典用于存储长度对应的reads信息

    for read_seq in seq_len:
        read_len_dict.setdefault(read_seq, 0)
        read_len_dict[read_seq] += 1
    # print(read_len_dict)
    return read_len_dict


def write_count(seq_id, seq, seq_len, read_len_dict):
    # pandas 写入序列编号、序列信息、序列长度
    data1 = pd.DataFrame({"Seq_ID": seq_id})
    data2 = pd.DataFrame({"Seq_Info": seq})
    data3 = pd.DataFrame({"Seq_Len": seq_len})
    # 统计不同长度序列的条数
    k_len = []
    v_count = []
    for k_len, v_count in read_len_dict.items():
        k_len.append()
        v_count.append()
        return k_len, v_count
    data4 = pd.DataFrame({"Read_Len": k_len})
    data5 = pd.DataFrame({"Read_Len": v_count})

    # 将序列编号，序列信息，以及序列长度写入*.xlsx文件中
    writer = pd.ExcelWriter(abs_path + "\\" + "test3.xlsx")
    data1.to_excel(writer, sheet_name="data", startcol=0, index=False)
    data2.to_excel(writer, sheet_name="data", startcol=1, index=False)
    data3.to_excel(writer, sheet_name="data", startcol=2, index=False)
    # 将序列统计后的数据写入 reads_len_count中
    data4.to_excel(writer, sheet_name="reads_len_count", startcol=0, index=False)
    data5.to_excel(writer, sheet_name="reads_len_count", startcol=1, index=False)
    writer.save()  # 数据保存为excel文件


def count_bar(seq_len):
    """
    根据上一步获得的序列长度信息，对其进行sort/uniq，matplotlib 处理并绘制直方图
    首先对数据进行排序统计对相同长度进行计数
    数据清洗后进行画bar图
    提取上一步获得第三列长度数据进行清洗并统计每个长度的个数
    """
    len_count = Counter(seq_len)
    # matplotlib绘图
    x = []
    y = []
    for k, v in len_count.items():
        x.append(k)
        y.append(v)
    # print(x, y)

    x_major_locator = MultipleLocator(200)
    y_major_locator = MultipleLocator(5)
    # 把x轴的刻度间隔设置为1，并存在变量里
    # y_major_locator=MultipleLocator(10)
    # 把y轴的刻度间隔设置为10，并存在变量里
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    # 把x轴的主刻度设置为1的倍数
    # ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    # plt.xlim(0,3000)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    # plt.ylim(-5,110)
    print(plt.xlim([0, 3000]))
    print(plt.ylim([0, 80]))

    plt.bar(x, y, color="rgb", lw=10)
    plt.xlabel("Sequence length")
    plt.ylabel("Count")
    # plt.savefig()
    # plt.tight_layout()
    plt.xticks(rotation=30)  # rotation控制倾斜角度
    plt.show()


def count_average(seq_len):
    len_count = Counter(seq_len)
    # matplotlib绘图
    x = []
    y = []
    sum = 0
    for k, v in len_count.items():
        sum += k * v
    print(sum * 1.0 / 14189)


if __name__ == "__main__":
    path = "data/"
    abs_path = os.getcwd()  # 获取当前目录路径
    # print(abs_path)
    # file_name = path+'DNA_Nopading_PDB14189' # 获取当前目录下的文件信息
    seq_id = []  # 新建列表存储fasta文件中序列编号信息
    seq = []  # 新建列表存储对应fasta文件中序列信息
    seq_len = []  # 新建列表存储对应序列的长度信息
    read_len_dict = {}
    read_fastq_seq(path + "DNA_Original_PDB14189")
    # write_count(seq_id, seq, seq_len,read_len_dict)
    count_reads_rate(seq, seq_len)
    count_bar(seq_len)
    count_average(seq_len)
