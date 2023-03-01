# tf.train.string_input_producer; 从文件中读取数据

import tensorflow as tf

filename = ['data/A.csv', 'data/B.csv', 'data/C.csv']

# 生成文件队列
file_queue = tf.train.string_input_producer(filename,
                                            shuffle=True,
                                            num_epochs=2)
reader = tf.WholeFileReader() # 文件读取器
key, value = reader.read(file_queue) # key为文件名tensor，value为文件内容tensor

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
    for i in range(6):
        print(sess.run([key, value]))