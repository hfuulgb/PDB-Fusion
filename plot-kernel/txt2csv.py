import xlwt
import os
import sys
import io
import pandas as pd

# 设置表格样式
def set_style(name, height, alignment, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


# 获取字符串长度，一个中文的长度为2
def len_byte(value):
    length = len(value)
    utf8_length = len(value.encode("utf-8"))
    length = (utf8_length - length) / 2 + length
    return int(length)


# txt转excel
def txt_xls(filename, xlsname):
    try:
        f = open(filename, "r", encoding="utf-8")  # 加b是二进制，需要加encoding，不然后面报错
        xls = xlwt.Workbook()
        # 生成excel的方法，声明excel
        sheet = xls.add_sheet("sheet", cell_overwrite_ok=True)
        result = []
        i = 0
        for line in f:  #
            item = line.split(",")
            result.append(item)
            for j in range(len(item)):
                sheet.write(i, j, item[j], set_style("Times New Roman", 220, "bottom"))
            i += 1

        col_width = []
        for i in range(len(result)):
            for j in range(len(result[i])):
                if i == 0:
                    col_width.append(len_byte(result[i][j]))
                else:
                    if col_width[j] < len_byte(str(result[i][j])):
                        col_width[j] = len_byte(result[i][j])

        # 设置栏位宽度，栏位宽度小于10时候采用默认宽度
        for i in range(len(col_width)):
            if col_width[i] > 10:
                sheet.col(i).width = 256 * (col_width[i] + 1)

        f.close()
        xls.save(xlsname)  # 保存为xls文件
        f.close()
    except:
        raise


if __name__ == "__main__":
    filename = "k7-cnn.txt"
    #    filename=filename.encode()
    xlsname = "k7-cnn.xls"
    txt_xls(filename, xlsname)
