# -*- coding: utf-8 -*-
__modifier__= 'jeongjae0815@gmail.com'

import tensorflow as tf
import matplotlib.pyplot as plt

#다양한 방법으로 변수 생성
bias_1=tf.Variable(tf.zeros(shape=[200]),name="b1")
w1=tf.Variable(tf.linspace(start=0.0,stop=12.0, num=3),name="w1")
w2=tf.Variable(tf.range(start=0.0, limit=12.0,delta=3),name="w2")
w3=tf.Variable(tf.random_normal(shape=[784,200],mean=1.5,stddev=0.35),name="w3")
w4=tf.Variable(tf.truncated_normal(shape=[784,200],mean=1.5,stddev=0.35),name="w4")

print w1
print w2
print w3
print w4

init_op=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    val_b1 = sess.run(bias_1)
    val_w1, val_w2, val_w3, val_w4 = sess.run([w1, w2, w3, w4])
    print(type(val_b1))
    print(val_w1.shape)

    # 그래프로 변수 확인하기
    plt.subplot(221)
    plt.hist(val_w1)
    plt.title('val_w1_linspace')
    plt.grid(True)

    plt.subplot(222)
    plt.hist(val_w2)
    plt.title('val_w2_range')
    plt.grid(True)

    plt.subplot(223)
    plt.hist(val_w3)
    plt.title('val_w3_random_normal')
    plt.grid(True)

    plt.subplot(224)
    plt.hist(val_w4)
    plt.title('val_w2_truncated_normal')
    plt.grid(True)

    #plt.show()

with tf.device("/cpu:0"):
    bias_2 = tf.Variable(tf.ones(shape=[200]), name="b2")
print bias_1
print bias_2


#저장
model_path="./tmp/tx-01.ckpt"

bias_3=tf.Variable(tf.add(bias_1,bias_2),name="b3")
init_op=tf.global_variables_initializer()

saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)

    val_b3=sess.run(bias_3)
    print val_b3
    save_path=saver.save(sess,model_path)
    print "model saved in file : %s" %save_path