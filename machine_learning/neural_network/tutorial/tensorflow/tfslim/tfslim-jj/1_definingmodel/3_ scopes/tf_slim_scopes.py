# -*- coding: utf-8 -*-
__modifier__='jeongjae0815@gmail.com'
import tensorflow as tf
import tensorflow.contrib.slim as slim

# 아래 코드는 padding이나 initalizer이 중복이 있어 가독성이 떨어진다
with tf.variable_scope('test1'):
    input_val = tf.placeholder(tf.float32, [16, 300, 300, 64])

    net1 = slim.conv2d(inputs=input_val, num_outputs=64, kernel_size=[11, 11], stride=4, padding='SAME',
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       weights_regularizer=slim.l2_regularizer(0.0005), scope='conv1')
    net1 = slim.conv2d(inputs=net1, num_outputs=128, kernel_size=[11, 11], padding='VALID',
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       weights_regularizer=slim.l2_regularizer(0.0005), scope='conv2')
    net1 = slim.conv2d(inputs=net1, num_outputs=256, kernel_size=[11, 11], padding='SAME',
                       weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       weights_regularizer=slim.l2_regularizer(0.0005), scope='conv3')

#그런데 위 코드에서 공통된  hyperparameter를 다음과 같이 묵을수 있다.
with tf.variable_scope('test2'):
    padding = 'SAME'
    initializer = tf.truncated_normal_initializer(stddev=0.01)
    regularizer = slim.l2_regularizer(0.0005)
    net2 = slim.conv2d(inputs=input_val, num_outputs=64, kernel_size=[11, 11], stride=4,
                       padding=padding,
                       weights_initializer=initializer,
                       weights_regularizer=regularizer,
                       scope='conv1')
    net2 = slim.conv2d(inputs=net2, num_outputs=128, kernel_size=[11, 11],
                       padding='VALID',
                       weights_initializer=initializer,
                       weights_regularizer=regularizer,
                       scope='conv2')
    net2 = slim.conv2d(inputs=net2, num_outputs=256, kernel_size=[11, 11],
                       padding=padding,
                       weights_initializer=initializer,
                       weights_regularizer=regularizer,
                       scope='conv3')
#하지만 여전히 뭔가 부족한 느낌. slim.arg_scop를 사용하여 hpyerparmeter를 arg_scope내에서 다시 정의할 수 있다
with tf.variable_scope('test3'):
    with slim.arg_scope([slim.conv2d], padding='SAME',
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net3=slim.conv2d(inputs=input_val,num_outputs=64,kernel_size=[11,11],stride=4,scope='conv1')
        net3 = slim.conv2d(inputs=net3, num_outputs=128, kernel_size=[11, 11], padding='VALID', scope='conv2')
        net3 = slim.conv2d(inputs=net3, num_outputs=256, kernel_size=[11, 11], scope='conv3')

model_variables = slim.get_model_variables()
print [var.name for var in model_variables]

#또는 arg_scope를 netting하거나 use multiple operations를 한 스코프안에서 할 수 있다
with tf.variable_scope('test5'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
            net4 = slim.conv2d(inputs=input_val, num_outputs=64, kernel_size=[11, 11], stride=4, scope='conv1')
            net4 = slim.conv2d(inputs=net4, num_outputs=256, kernel_size=[5, 5],
                               weights_initializer=tf.truncated_normal_initializer(stddev=0.03),
                               scope='conv2')
            net4 = slim.fully_connected(inputs=net4, num_outputs=1000, activation_fn=None, scope='fc')