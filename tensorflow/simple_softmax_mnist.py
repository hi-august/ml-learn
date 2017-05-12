# coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

# wasn't compiled to use SSE报错
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 会自动从网上下载测试数据
#  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # 默认下载到/tmp目录
#  print("Download Done!")

# 因为网络原因这里设定为下载了的目录
FLAGS = None
flags = tf.app.flags
FLAGS = flags.FLAGS
# 定制读取mnist目录
flags.DEFINE_string('data_dir', os.getcwd()+'/data', 'custom store mnist data')
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# 这是占位符,数据类型是float,
# x占位符形状是[None, 784],这个是用来存放图像数据的变量,由28*28(长和宽像素)得来
# y_占位符形状为[None, 10],输出结果(0-9),所以维度只有10
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 定义模型权重W和偏置b
# 初始化为零向量,W是一个784*10的矩阵(因为有784个特征和十个输出值)
# b是一个10维的向量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax regression是一个简单的模型
# 用来处理一个待分类对象在多个类别上的概率分布
# 这一般是很多高级模型的最后一步
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss func
# 在机器学习中,通常要选择一个代价函数(损失函数),来指示训练模型的好坏
# 这里使用交叉熵函数(cross-entropy)作为代价函数
# 交叉熵函数的输出值表示预测的概率分布与真是的分布的符合程度
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 梯度下降算法
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# 开始训练模型
for i in range(1000):
    # 随机取出100数据执行train_step
    # 这个过程称为随机梯度下降
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 评价模型(准确率)
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("Accuarcy on Test-dataset: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
