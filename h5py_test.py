import h5py
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

# f = h5py.File("mytestfile.hdf5", "w")
# dset = f.create_dataset("mydataset", (100,), dtype='i')

f = h5py.File("mytestfile1.hdf5", "w")

# 遍历Video_pic文件夹及其子文件夹
root_dir = "./UCF_Crime_Frames/"

def create_group(path):
    # 在HDF5文件中创建与文件夹相对应的组
    group_name = os.path.relpath(path, root_dir)
    if group_name == '.':
        return f
    else:
        return f.require_group(group_name)

for root, dirs, files in os.walk(root_dir):
    # 对于每个文件夹，创建相应的HDF5组
    # h5_group = create_group(root)

    # 遍历文件夹中的图片文件
    for file in tqdm(files):
        if file.endswith(".jpg") or file.endswith(".png"):
            # 读取图像文件
            image_path = os.path.join(root, file)
            image = Image.open(image_path)
            image_array = np.array(image)

            # 将图像数据写入HDF5文件
            dataset_name = os.path.join(os.path.relpath(root, root_dir), file)
            f.create_dataset(dataset_name, data=image_array)

# 关闭HDF5文件
f.close()


