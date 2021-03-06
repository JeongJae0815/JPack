# -*- coding: utf-8 -*-
__modifier__='jeongjae0815@gmail.com'

import tensorflow as tf
import tensorflow.contrib.slim as slim

"""
native TF에서 layer 정의하는 방법
"""
input_val = tf.placeholder(tf.float32, [16, 32, 32, 64])
with tf.variable_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1, name='weight'))
    conv = tf.nn.conv2d(input=input_val, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name='activation')

"""
slim 에서 layer 정의하는 방법
"""

#padding='SAME' is default
#stride=[1,1,1,1] si defualt

net=slim.conv2d(inputs=input_val,num_outputs=128,kernel_size=[3,3],scope='conv1_1')

"""
그리고 slim에서는 두가지 meata-opreation을 제공하는데, repeat와 stack이라는 것을 지원 한다.
"""

#the following snippet from the VGG network
net1 = tf.placeholder(tf.float32, [16, 32, 32, 256])
with tf.variable_scope('test1') as scope:
    net1 = slim.conv2d(net1, 256, [3, 3], scope='conv3_1')
    net1 = slim.conv2d(net1, 256, [3, 3], scope='conv3_2')
    net1 = slim.conv2d(net1, 256, [3, 3], scope='conv3_3')
    net1 = slim.max_pool2d(net1, [2, 2], scope='pool2')
model_variables = slim.get_model_variables()
#print [var.name for var in model_variables]

#for loop 사용
net2=tf.placeholder(tf.float32,[16,32,32,256])
with tf.variable_scope('test2')as scope:
    for i in range(3):
        net=slim.conv2d(net2,256,[3,3],scope='conv3_%d'%(i+1))
    net2=slim.max_pool2d(net2,[2,2],scope='pool2')

model_variables = slim.get_model_variables()
#print [var.name for var in model_variables]

#repeat 사용
net3=tf.placeholder(tf.float32,[16,32,32,256])
with tf.variable_scope('test3') as scope:
    net3 = slim.repeat(net3,3,slim.conv2d,256,[3,3],scope='conv3')
    net3 = slim.max_pool2d(net2,[2,2],scope='pool2')

model_variables = slim.get_model_variables()
#print [var.name for var in model_variables]


#stack 사용 예
#MLP 일부
g=tf.Graph()
with g.as_default():
    input_val=tf.placeholder(tf.float32, [16,4])
    mlp1 = slim.fully_connected(inputs=input_val, num_outputs=32, scope='fc/fc_1')
    mlp1 = slim.fully_connected(inputs=mlp1, num_outputs=64, scope='fc/fc_2')
    mlp1 = slim.fully_connected(inputs=mlp1, num_outputs=64, scope='fc/fc_3')

    train_writer = tf.summary.FileWriter('./tmp/tf-slim-tutorial1', g)
    train_writer.close()

print [node.name for node in g.as_graph_def().node]

#stack 사용
g=tf.Graph()
with g.as_default():
    input_val=tf.placeholder(tf.float32, [16,4])
    mlp2=slim.stack(input_val,slim.fully_connected,[32,64,128],scope='fc')
    print [node.name for node in g.as_graph_def().node]


    train_writer = tf.summary.FileWriter('./tmp/tf-slim-tutorial2', g)
    train_writer.close()

g=tf.Graph()
with g.as_default():
    input_val = tf.placeholder(tf.float32, [16,32,32,8])
    conv2=slim.stack(input_val,slim.conv2d,[(32, [3,3]),(32, [1,1]),(64, [3,3]),(64, [1,1])],scope='core')
    print [node.name for node in g.as_graph_def().node]