# -*- coding: utf-8 -*-
__author__ = 'socurites@gmail.com'

"""
TFRecord Dataset 로드하기
# 1. TFRecord 포맷 데이터을 읽어서 변환할 수 있도록 slim.dataset.Dataset 클래스를 정의한다.
# 2. 데이터를 피드하기 위한 slim.dataset_data_provider.DatasetDataProvider를 생성한다.
# 3. 네트워크 모델의 입력에 맞게 전처리 작업 및 편의를 위한 one-hot 인코딩 작업을 한 후, tf.train.batch를 생성한다.
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
import load_tfrecord
#from utils.dataset_utils import load_batch

"""
# slim.dataset.Dataset 클래스를 정의
"""
TF_RECORD_DIR = './flowers/tfrecord'
mnist_tfrecord_dataset = load_tfrecord.TFRecordDataset(tfrecord_dir=TF_RECORD_DIR,
                                                           dataset_name='mnist',
                                                           num_classes=17)
# train 데이터셋 생성
dataset = mnist_tfrecord_dataset.get_split(split_name='train')
images,label,_ = load_tfrecord.load_batch(dataset)
import matplotlib.pyplot as plt


with tf.Session() as sess:
    with slim.queues.QueueRunners(sess):
        imgs, lbls = sess.run([images, label])
        writer = tf.summary.FileWriter('./tmp/tf-slim-tutorial')
        writer.add_graph(sess.graph)

        print lbls

