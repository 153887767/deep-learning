# tf.train.slice_input_producer; 读取文件列表中的样本

import tensorflow as tf

images = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg']
labels = [1, 2, 3, 4]

# 获取图片数据张量(tensor)和图片label张量
[images, labels] = tf.train.slice_input_producer([images, labels],
                              num_epochs=None, # epoch：1个epoch等于使用训练集中的全部样本训练一次
                              shuffle=True)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer()) # 变量初始化(num_epochs, shuffle)

    tf.train.start_queue_runners(sess=sess) # 开启文件队列填充线程

    for i in range(10):
        print(sess.run([images, labels])) # 从文件队列获取数据，并打印
