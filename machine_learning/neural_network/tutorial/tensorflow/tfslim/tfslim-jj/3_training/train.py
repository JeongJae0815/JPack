# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim.nets as net
import tensorflow.contrib.slim as slim
import load_tfrecord

def vgg16(inputs):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')
    net = slim.fully_connected(net, 4096, scope='fc6')
    net = slim.dropout(net, 0.5, scope='dropout6')
    net = slim.fully_connected(net, 4096, scope='fc7')
    net = slim.dropout(net, 0.5, scope='dropout7')
    net = slim.fully_connected(net, 17, activation_fn=None, scope='fc8')
  return net








TF_RECORD_DIR = '/home/jeongjaepark/NeuralNetwork/Python_functions_compliations/tesorflow/tfslim-jj/2_loading_datasets/flowers/tfrecord'
flowers_tfrecord_dataset = load_tfrecord.TFRecordDataset(tfrecord_dir=TF_RECORD_DIR,
                                                           dataset_name='mnist',
                                                           num_classes=17)
dataset = flowers_tfrecord_dataset.get_split(split_name='train')
images,label,_ = load_tfrecord.load_batch(dataset)

X=tf.placeholder(tf.float32,[None,224,224,3])
with slim.arg_scope(vgg.vgg_arg_scope()):
    logits,_=vgg.vgg_16(X,num_classes=17,is_training=True)
init_op = tf.global_variables_initializer()
print logits

with tf.Session() as sess:
    sess.run(init_op)

    with slim.queues.QueueRunners(sess):
      imgs=sess.run(images)
      lg = sess.run(logits, feed_dict={X: imgs})


#loss function
loss=slim.losses.softmax_cross_entropy(logits=logits,onehot_labels=label)
total_loss= slim.losses.get_total_loss()

optimizer=tf.train.AdamOptimizer(learning_rate=0.001)

predictions = tf.argmax(logits,1)
targets = tf.arg_max(label,1)

correct_prediction=tf.equal(predictions,targets)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tf.summary.scalar('losses/Total',total_loss)
tf.summary.scalar('accuracy',accuracy)
summar_op=tf.summary.merge_all()

log_dir='./tmp/tfslim_model'
if not tf.gfile.Exists(log_dir):
    tf.gfile.MakeDirs(log_dir)

train_op = slim.learning.create_train_op(total_loss, optimizer)

final_loss=slim.learning.train(
    train_op,
    log_dir,
    number_of_steps=2000,
    summary_op=summar_op,
    save_summaries_secs=30,
    save_interval_secs=30
)