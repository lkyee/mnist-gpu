import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np



def inference(input_tensor, train,regularizer):
    with tf.device("/cpu:0"):
        with tf.variable_scope('layer1-conv1'):  # 声明第一层神经网络的变盘并完成前向传播过程。
            conv1_weights = tf.get_variable("weight", [5, 5, 1, 32],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

        with tf.name_scope('layer2_pool1'):
            pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope('layer3_conv2'):
            conv2_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

        with tf.name_scope('layer4_pool2'):
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        pool_shape = pool2.get_shape().as_list()  # 返回[]的shape值，如果去掉.as_list()则返回的tuple
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]  # pool_shape[0]为一个batch中数据的个数
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

        with tf.variable_scope('layer5_fc1'):
            fc1_weights = tf.get_variable("weight", [nodes, 512],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None:
                tf.add_to_collection("losses", regularizer(fc1_weights))  # 只有全连接层的权重要加入正则化~
            fc1_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.0))
            fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
            if train: fc1 = tf.nn.dropout(fc1, 0.5)  # dropout一般只在全连接层使用

        with tf.variable_scope('layer6_fc2'):
            fc2_weights = tf.get_variable("weight", [512, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))

            if regularizer != None:
                tf.add_to_collection("losses", regularizer(fc2_weights))  # 只有全连接层的权重要加入正则化
            fc2_biases = tf.get_variable("bias", [10], initializer=tf.constant_initializer(0.0))
            logit = tf.matmul(fc1, fc2_weights) + fc2_biases
        return logit




def train(mnist):
    x = tf.placeholder(tf.float32, [128,28,28,1],name='x_input')
    y_gt = tf.placeholder(tf.float32, [None, 10], name='y_gt')
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    y_prd = inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.99, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_prd, labels=tf.argmax(y_gt, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))  # 加上正则化的项
    learning_rate = tf.train.exponential_decay(0.01,global_step,mnist.train.num_examples / 128, 0.99,staircase=True)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_op=tf.group(train_step,variables_averages_op)

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)) as sess:   #创建会话
        tf.global_variables_initializer().run()
        for i in range(30000):
            xs, ys = mnist.train.next_batch(128)#产生这一轮使用的一个batch 的训练数据
            reshape_xs=np.reshape(xs,(128,28,28,1))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshape_xs, y_gt: ys})
            if i % 10 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
            if i%1000==0:
                saver.save(sess, os.path.join("model/", "mnist_model"), global_step=global_step)
                #saver.save(sess,save_path,global_step)   用于保存模型


def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':   #如果py文件被直接运行则该代码块将被运行，当py文件以模块的方式导入时，代码块将不被运行
    tf.app.run()    #处理flag解析，然后执行main函数



'''
1.缩进问题
2.logit和label搞错
3.学习率设置过大
？：为什么设置成None后报错
import tensorflow as tf
a=tf.constant([1.0,2.0,3.0],shape=[3],name='a')
b=tf.constant([1.0,2.0,3.0],shape=[3],name='b')
c=a+b
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))

import tensorflow as tf
with tf.device('/gpu:0'):
    a=tf.constant([1.0,2.0,3.0],shape=[3],name='a')
    b=tf.constant([1.0,2.0,3.0],shape=[3],name='b')
    c=a+b
sess=tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True))
print(sess.run(c))
'''