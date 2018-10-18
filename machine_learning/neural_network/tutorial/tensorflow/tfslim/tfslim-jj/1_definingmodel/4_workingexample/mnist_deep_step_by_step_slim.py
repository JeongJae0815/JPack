__author__ = 'socurites@gmail.com'
#minist slim
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
vgg=nets.vgg

def mnist_convnet(inputs, is_training=True):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1)):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=5):
            net = slim.conv2d(inputs=inputs, num_outputs=32, scope='conv1')
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool1')
            net = slim.conv2d(inputs=net, num_outputs=64, scope='conv2')
            net = slim.max_pool2d(inputs=net, kernel_size=[2, 2], scope='pool2')
            net = slim.flatten(inputs=net, scope='flatten')
            net = slim.fully_connected(inputs=net, num_outputs=1024, scope='fc3')
            net = slim.dropout(inputs=net, is_training=is_training, keep_prob=0.5, scope='dropout4')
            net = slim.fully_connected(inputs=net, num_outputs=10, activation_fn=None, scope='fc4')
    return net
# create variable for input and real output y
x = tf.placeholder(tf.float32, shape=[None, 784])
x = tf.placeholder(tf.float32, shape=[None, 600,512,3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# reshape x
x_image = tf.reshape(x, [-1, 28, 28, 1])


with tf.variable_scope('minist_layer_JJ'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
        with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
            net1 = slim.conv2d(inputs=x_image, num_outputs=32, kernel_size=[11, 11], activation_fn=None,stride=4, scope='conv1')
            net1 = slim.max_pool2d(net1,[2,2],padding='SAME',scope='pool1')
            net1 = slim.dropout(net1)

            net2 = slim.stack(net1, slim.conv2d, [(64, [5, 5]), (64, [5, 5]), (64, [5, 5])], scope='conv2')
            net2 = slim.max_pool2d(net2, [2, 2], padding='SAME',scope='pool2')
            net2 = slim.flatten(inputs=net2, scope='flatten')

            net3 = slim.stack(net2, slim.fully_connected,[1024,100,10],scope='fc')
            print net3

with tf.variable_scope('minist_layer'):
    net4=mnist_convnet(x_image)



with tf.Session() as sess:
    writer = tf.summary.FileWriter('./tmp/tf-slim-tutorial3', sess.graph)
    sess.run(tf.global_variables_initializer())
    writer.close()



