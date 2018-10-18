# -*- coding: utf-8 -*-
__modifier__= 'jeongjae0815@gmail.com'

import tensorflow as tf
import matplotlib.pyplot as plt

bias_3=tf.Variable(name="b3")
tf.reset_default_graph()
model_path = "./tmp/tx-01.ckpt"
# 로드
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, model_path)
    print("Model restored")

