import os

# 根据文件路径（相对或绝对）创建目录
def mkdirs(file):
    dir=os.path.dirname(os.path.abspath(file))
    print(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

