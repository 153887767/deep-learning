# 将图片数据打包成tfrecord二进制文件，这些数据将用于后续模型的训练

import tensorflow as tf
import cv2
import numpy as np
import glob

classification = ['airplane',
                  'automobile',
                  'bird',
                  'cat',
                  'deer',
                  'dog',
                  'frog',
                  'horse',
                  'ship',
                  'truck']

idx = 0
im_data = []
im_labels = []
for path in classification:
    # path = "data/image/train/" + path # 文件夹路径
    path = "data/image/test/" + path
    im_list = glob.glob(path + "/*") # 获取当前文件夹下的所有图片
    im_label = [idx for i in range(im_list.__len__())] # 长度为图片数量的list, 值全为idx
    idx += 1
    im_data += im_list
    im_labels += im_label

print(im_labels)
print(im_data)

# tfrecord_file = "data/train.tfrecord"
tfrecord_file = "data/test.tfrecord"
writer = tf.python_io.TFRecordWriter(tfrecord_file)

index = [i for i in range(im_data.__len__())]
np.random.shuffle(index) # 打乱索引

for i in range(im_data.__len__()):
    im_d = im_data[index[i]] # 乱序
    im_l = im_labels[index[i]]
    data = cv2.imread(im_d) # 读取图片数据
    ex = tf.train.Example(
        features = tf.train.Features(
            feature = {
                "image":tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[data.tobytes()])),
                "label": tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[im_l])),
            }
        )
    )
    writer.write(ex.SerializeToString()) # 数据序列化后写入

writer.close()
