import os
import shutil

for i in os.listdir('webface_unmasked/'):
    # 创建对应文件夹
    os.makedirs('webface_merged/' + i, 0o777)
    src = "./webface_unmasked/" + i  # 原文件夹路径
    des = "./webface_merged/" + i  # 目标文件夹路径

    for file in os.listdir(src):
        # 遍历原文件夹中的文件
        full_file_name = os.path.join(src, file)  # 把文件的完整路径得到
        print("要被复制的全文件路径全名:", full_file_name)
        if os.path.isfile(full_file_name):  # 用于判断某一对象(需提供绝对路径)是否为文件
            shutil.copy(full_file_name, des)  # shutil.copy函数放入原文件的路径文件全名  然后放入目标文件夹
