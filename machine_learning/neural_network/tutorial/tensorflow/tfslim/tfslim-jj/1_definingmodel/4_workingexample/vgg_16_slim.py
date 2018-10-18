import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import tensorflow.contrib.slim as slim
vgg=nets.vgg
x = tf.placeholder(tf.float32, shape=[None, 500,500,3])
y = tf.placeholder(tf.float32, shape=[None, 500,500,1])

with slim.arg_scope(vgg.vgg_arg_scope()):
  logits, end_points = vgg.vgg_16(inputs=x, num_classes=100, is_training=True,spatial_squeeze=False)


with tf.Session() as sess:
    writer = tf.summary.FileWriter('./tmp/vgg_16_slim', sess.graph)
    sess.run(tf.global_variables_initializer())
    writer.close()

