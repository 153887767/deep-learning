# 2-训练模型

import tensorflow as tf
import readcifar10
import os
# import resnet

# slim是对tensorflow的高级封装，搭建网络时代码简洁，可读性更高
slim = tf.contrib.slim

# 模型，定义网络结构
def model(image, keep_prob=0.8, is_training=True):

    # batch_norm参数
    batch_norm_params = {
        "is_training": is_training, # 训练时为true, 测试时为false
        "epsilon":1e-5, # 防止batch_norm归一化时除0
        "decay":0.997, # 衰减系数
        'scale':True,
        'updates_collections':tf.GraphKeys.UPDATE_OPS # 收集batchNorm层参数
    }

    # 卷积层和池化层默认参数初始化
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer = slim.variance_scaling_initializer(), # 卷积层初始化参数
        activation_fn = tf.nn.relu, # 卷积层激活方式，卷积之后加入激活函数，使用relu激活函数，f(x)=max(0,x)
        weights_regularizer = slim.l2_regularizer(0.0001), # 权值正则约束，采用l2正则，正则项权值为0.0001
        normalizer_fn = slim.batch_norm, # 把每层神经网络任意神经元输入值的分布强行拉回到均值为0方差为1的标准正态分布
        normalizer_params = batch_norm_params): # batch_norm参数
        with slim.arg_scope([slim.max_pool2d], padding="SAME"):
            # 利用卷积层、池化层、全连接层、dropout层搭建用于cifar10图像分类的卷积神经网，参照VGGNet思路
            net = slim.conv2d(image, 32, [3, 3], scope='conv1') # 第一个卷积层，32个卷积核，卷积核大小为3*3，卷积层命名为conv1
            net = slim.conv2d(net, 32, [3, 3], scope='conv2') # 第二个卷积层
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1') # 池化层(max-pool层)
            net = slim.conv2d(net, 64, [3, 3], scope='conv3')
            net = slim.conv2d(net, 64, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2')
            net = slim.conv2d(net, 128, [3, 3], scope='conv5')
            net = slim.conv2d(net, 128, [3, 3], scope='conv6')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool3')
            net = slim.conv2d(net, 256, [3, 3], scope='conv7')
            net = tf.reduce_mean(net, axis=[1, 2]) # 特征图维度为nhwc，对特征值求均值得到n11c四个维度的特征图
            net = slim.flatten(net) # 转为c维
            net = slim.fully_connected(net, 1024) # 全连接层1
            slim.dropout(net, keep_prob) # dropout层，keep_prob是概率值，训练时keep_prob<1，测试时keep_prob=1
            net = slim.fully_connected(net, 10) # 全连接层2，10个维度的概率分布
    return net # 返回一个10维向量

# 损失层
def loss(logits, label):
    # 分类损失
    one_hot_label = slim.one_hot_encoding(label, 10) # one_hot编码，onehot长度定义为10
    slim.losses.softmax_cross_entropy(logits, one_hot_label) # 交叉熵损失，logits是预测出来的概率分布值

    # 正则化损失
    reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) # 获取正则化loss
    l2_loss = tf.add_n(reg_set) # 计算出整体的l2损失
    slim.losses.add_loss(l2_loss) # 把l2损失添加到loss中

    totalloss = slim.losses.get_total_loss() # 计算出所有loss

    return totalloss, l2_loss

# 优化器
def func_optimal(batchsize, loss_val):
    global_step = tf.Variable(0, trainable=False)
    # 定义学习率变化
    lr = tf.train.exponential_decay(0.01, # 初始学习率
                                    global_step, # 当前迭代次数，从0开始
                                    decay_steps= 50000// batchsize, # 每隔decay_steps步，对学习率进行一次衰减
                                    decay_rate= 0.95, # 每次衰减比例
                                    staircase=False) # 学习率平滑下降
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # 收集batchNorm层参数

    with tf.control_dependencies(update_ops):
        op = tf.train.AdamOptimizer(lr).minimize(loss_val, global_step) # 优化器，minimize最小化损失
    return global_step, op, lr # 当前迭代次数，调节网络参数的优化器，学习率

# 训练
def train():
    batchsize = 64
    floder_log = 'logdirs' # 日志存放的目录
    # floder_log = 'logdirs-resnet'
    floder_model = 'model' # 模型存放的目录
    # floder_model = 'model-resnet'

    if not os.path.exists(floder_log):
        os.mkdir(floder_log)

    if not os.path.exists(floder_model):
        os.mkdir(floder_model)

    tr_summary = set() # 存放训练样本的日志信息
    te_summary = set() # 存放测试样本的日志信息

    # 数据
    tr_im, tr_label = readcifar10.read(batchsize, 0, 1) # 训练样本
    te_im, te_label = readcifar10.read(batchsize, 1, 0) # 测试样本

    # 网络
    input_data = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='input_data') # 输入数据，32*32*3，3指3通道(rgb)
    input_label = tf.placeholder(tf.int64, shape=[None], name='input_label')
    keep_prob = tf.placeholder(tf.float32, shape=None, name='keep_prob') # dropout层概率值
    is_training = tf.placeholder(tf.bool, shape=None, name='is_training')
    logits = model(input_data, keep_prob=keep_prob, is_training=is_training) # 得到输出结果
    # logits = resnet.model_resnet(input_data, keep_prob=keep_prob, is_training=is_training) # resnet

    # 损失
    total_loss, l2_loss = loss(logits, input_label) # 传入网络预测结果和label
    tr_summary.add(tf.summary.scalar('train total loss', total_loss))
    tr_summary.add(tf.summary.scalar('test l2_loss', l2_loss))
    te_summary.add(tf.summary.scalar('train total loss', total_loss))
    te_summary.add(tf.summary.scalar('test l2_loss', l2_loss))

    # 精度
    pred_max  = tf.argmax(logits, 1) # 当前概率分布最大的值对应的索引
    correct = tf.equal(pred_max, input_label) # 判断pred_max和input_label是否相等
    accurancy = tf.reduce_mean(tf.cast(correct, tf.float32)) # 定义精度，对correct进行统计
    tr_summary.add(tf.summary.scalar('train accurancy', accurancy))
    te_summary.add(tf.summary.scalar('test accurancy', accurancy))

    # 调用优化器
    global_step, op, lr = func_optimal(batchsize, total_loss)
    tr_summary.add(tf.summary.scalar('train lr', lr)) # 学习率
    te_summary.add(tf.summary.scalar('test lr', lr))
    tr_summary.add(tf.summary.image('train image', input_data * 128 + 128))
    te_summary.add(tf.summary.image('test image', input_data * 128 + 128))

    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer())) # 全局变量和局部变量初始化

        # 文件队列写入线程，启动多线程管理器
        tf.train.start_queue_runners(sess=sess,
                                     coord=tf.train.Coordinator())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5) # 用于模型存储，最大存放个数设置为5

        ckpt = tf.train.latest_checkpoint(floder_model) # 获取模型文件中最新的model(根据model/checkpoint文件查找)

        if ckpt:
            saver.restore(sess, ckpt) # 如果存在最新的model, 对当前这个网络进行恢复

        epoch_val = 100

        # 合并日志信息
        tr_summary_op = tf.summary.merge(list(tr_summary))
        te_summary_op = tf.summary.merge(list(te_summary))

        summary_writer = tf.summary.FileWriter(floder_log, sess.graph)

        for i in range(50000 * epoch_val): # 样本总量*epoch_val
            train_im_batch, train_label_batch = sess.run([tr_im, tr_label])
            feed_dict = {
                input_data:train_im_batch,
                input_label:train_label_batch,
                keep_prob:0.8,
                is_training:True
            }

            _, global_step_val, \
            lr_val, \
            total_loss_val, \
            accurancy_val, tr_summary_str = sess.run([op,
                                      global_step,
                                      lr,
                                      total_loss,
                                      accurancy, tr_summary_op],
                     feed_dict=feed_dict)

            # 把日志信息写入日志文件
            summary_writer.add_summary(tr_summary_str, global_step_val)

            # 打印训练样本效果
            if i % 100 == 0:
                print("{},{},{},{}".format(global_step_val,
                                           lr_val, total_loss_val,
                                           accurancy_val))

            # 打印测试样本效果
            # if i % (50000 // batchsize) == 0:
            if i % 100 == 0:
                test_loss = 0
                test_acc = 0
                # for ii in range(10000//batchsize):
                for ii in range(1):
                    test_im_batch, test_label_batch = \
                        sess.run([te_im, te_label])
                    feed_dict = {
                        input_data: test_im_batch,
                        input_label: test_label_batch,
                        keep_prob: 1.0,
                        is_training: False
                    }

                    total_loss_val, global_step_val, \
                    accurancy_val, te_summary_str = sess.run([total_loss,global_step,
                                              accurancy, te_summary_op],
                                             feed_dict=feed_dict)

                    summary_writer.add_summary(te_summary_str, global_step_val)

                    test_loss += total_loss_val
                    test_acc += accurancy_val

                # print('test：', test_loss * batchsize / 10000,
                #       test_acc* batchsize / 10000) # 测试集上的平均loss和精度
                print('test：', test_loss, test_acc)

            if i % 500 == 0:
                saver.save(sess, "{}/model.ckpt{}".format(floder_model, str(global_step_val))) # 模型存储
    return

# 当文件当作脚本运行时候，就执行代码；当文件被当做Module被import的时候，就不执行相关代码
if __name__ == '__main__':
    train()

# 查看log日志命令：tensorboard --logdir=logdirs
# 然后访问 http://localhost:6006
