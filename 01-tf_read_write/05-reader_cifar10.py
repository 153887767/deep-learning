# 解析tfrecord打包的数据

import tensorflow as tf

filelist = ['data/train.tfrecord']

# 文件队列
file_queue = tf.train.string_input_producer(filelist,
                                            num_epochs=None,
                                            shuffle=True)
reader = tf.TFRecordReader()
_, ex = reader.read(file_queue) # 读取数据，之前打包的时候对数据做了序列化，所以还需要解码

# 序列化的格式
feature = {
    'image':tf.FixedLenFeature([], tf.string),
    'label':tf.FixedLenFeature([], tf.int64)
}

# batchsize：批大小。在深度学习中，一般采用SGD训练，即每次训练在训练集中取batchsize个样本训练
# capacity: 队列容量
# min_after_dequeue: 最小队列容量
batchsize = 2
# 返回一个batchsize的数据
batch = tf.train.shuffle_batch([ex], batchsize, capacity=batchsize*10,
                       min_after_dequeue=batchsize*5)

# 解码，指定解码格式
example = tf.parse_example(batch, features=feature)

image = example['image']
label = example['label']

image = tf.decode_raw(image, tf.uint8) # 把字符串解码为unit8格式的数据
image = tf.reshape(image, [-1, 32, 32, 3]) # 列表第一个值填batchsize或-1, -1自动计算

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners(sess=sess) # 文件填充队列线程

    for i in range(1):
        image_bth, label_bth = sess.run([image,label]) # 从队列读取数据
        print(label_bth)
        import cv2
        cv2.imshow("image", image_bth[0,...]) # 可视化第一张图
        cv2.waitKey(0) # 等待
