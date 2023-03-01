# 1-数据读取

import tensorflow as tf

def read(batchsize=64, type=1, no_aug_data=1):
    reader = tf.TFRecordReader()

    if type == 0:
        file_list = ["data/train.tfrecord"] # 训练集
    if type == 1:
        file_list = ["data/test.tfrecord"] # 测试集

    # 文件名队列
    filename_queue = tf.train.string_input_producer(
        file_list, num_epochs=None, shuffle=True
    )
    _, serialized_example = reader.read(filename_queue)

    # 一个batchsize数据
    batch = tf.train.shuffle_batch([serialized_example], batchsize, capacity=batchsize * 10,
                                   min_after_dequeue= batchsize * 5)

    # 序列化的格式
    feature = {'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)}

    # 解码，指定解码格式
    features = tf.parse_example(batch, features = feature)

    images = features["image"]

    img_batch = tf.decode_raw(images, tf.uint8) # 把字符串解码为unit8格式的数据
    img_batch = tf.cast(img_batch, tf.float32) # 数据类型转换，转为float32
    img_batch = tf.reshape(img_batch, [batchsize, 32, 32, 3]) # 调整矩阵维度

    # 数据增强，对图像造成扰动，产生更多样本，提高模型泛化能力
    if type == 0 and no_aug_data == 1:
        # 随机裁剪
        distorted_image = tf.random_crop(img_batch,
                                         [batchsize, 28, 28, 3])
        # 随机对比度
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.8,
                                                   upper=1.2)
        # 随机色调
        distorted_image = tf.image.random_hue(distorted_image,
                                              max_delta=0.2)
        # 随机饱和度
        distorted_image = tf.image.random_saturation(distorted_image,
                                                     lower=0.8,
                                                     upper=1.2)
        img_batch = tf.clip_by_value(distorted_image, 0, 255) # 将张量中的数值限制在一个范围之内

    img_batch = tf.image.resize_images(img_batch, [32, 32]) # 由于存在裁剪，重新转为32*32
    label_batch = tf.cast(features['label'], tf.int64) # 数据类型转换，转为int64
    img_batch = tf.cast(img_batch, tf.float32) / 128.0 - 1.0 # 归一化，-1~1之间（原始数据为0~255）
    
    return img_batch, label_batch
