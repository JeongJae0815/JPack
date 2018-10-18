# -*- coding: utf-8 -*-
__modifier__= 'jeongjae0815@gmail.com'

import tensorflow as tf
import tensorflow.contrib.slim as slim

#일단 변수  설정하는 것은 native tensorflow와 거의 같다
#with tf.device("/cpu:0"):
#    w4=tf.Variable(tf.truncated_normal(shape=[784,200],mean=1.5, stddev=0.35),name"w4")

w4=slim.variable('w4',shape=[784,200],initializer=tf.truncated_normal_initializer(mean=1.5,stddev=0.35),device='/CPU:0')
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    val_w4=sess.run(w4)

#모델 변수와 일반변수의 차이
# 모델 변수는 trainable, 일반변수는 untrainable

# Model variable
weight_5 = slim.model_variable('w5',
                               shape=[10, 10, 3, 3],
                               initializer=tf.truncated_normal_initializer(stddev=0.1),
                               regularizer=slim.l2_regularizer(0.05),
                               device='/CPU:0')

model_variables = slim.get_model_variables()
print [var.name for var in model_variables]

#regular variable
my_var_1 = slim.variable('mv1',
                         shape=[20, 1],
                         initializer=tf.zeros_initializer())

model_variables = slim.get_model_variables()
all_variables = slim.get_variables()

print [var.name for var in model_variables]
print [var.name for var in all_variables]