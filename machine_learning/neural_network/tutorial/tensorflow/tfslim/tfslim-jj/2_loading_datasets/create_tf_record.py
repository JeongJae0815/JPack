# -*- coding: utf-8 -*-
__author__ = 'jeongjae0815@gmail.com'

import tensorflow as tf
import tensorflow.contrib.slim as slim

import math
import os
import random
import sys
import io
import PIL
import build_data
import numpy as np
import matplotlib.pyplot as plt

"""
# MNITS의 경우
raw_data/
  |- mnist/
       |- images/
           |- 0/
           |- 1/
           |- 2/
           |- 3/
           |-- ...
"""

# 아래 코드는 flowers/ 데이터를 다운로드 후 TFRecord로 변환하는 코드임
# 아래 코드를 약간 변형해서 사용
# https://github.com/tensorflow/models/blob/master/slim/download_and_convert_data.py

FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'data_dir',
    './flowers/jpg',
    'this is flowers path of flowers training set'
)
tf.app.flags.DEFINE_string(
    'output_dir',
    './flowers/tfrecord',
    'this is flowers path of flowers training set'
)

tf.app.flags.DEFINE_string(
    'train_list',
    './flowers/train_list.txt',
    'train list'
)

_NUM_SHARDS = 4
_first=1
tf.app.flags.DEFINE_integer(
    'num_shards',
    5,
    'A number of sharding for TFRecord files(integer')

def _convert_datatset(dataset_split,dataset_dir,list):
    tmp= tf.gfile.Glob(os.path.join(dataset_dir, '*.jpg'))
    #load data_list
    img_names=[]
    labels=[]
    text_file = open(list, 'r')
    lines=text_file.readlines()
    random.shuffle(lines)
    for f in lines:
        basename=f.split(' ')[0]
        img_names.append(os.path.join(dataset_dir,basename))
        labels.append(int(f.split(' ')[1]))
    num_images = len(img_names)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))
    image_reader=build_data.ImageReader('jpeg',channels=3)
    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(
            FLAGS.output_dir,
            '%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id, _NUM_SHARDS))

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)

            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_images, shard_id))
                sys.stdout.flush()

                #Read the image.
                image_filename = img_names[i]
                image_data = tf.gfile.FastGFile(image_filename, 'r').read()
                height, width = image_reader.read_image_dims(image_data)
                example = build_data.image_seg_to_tfexample(
                    image_data, img_names[i], height, width, labels[i])
                tfrecord_writer.write(example.SerializeToString())
            sys.stdout.write('\n')
            sys.stdout.flush()

def main(unused_argv):
    tf.gfile.MakeDirs(FLAGS.output_dir)

    _convert_datatset(
        'train', FLAGS.data_dir,FLAGS.train_list)
if __name__== '__main__':
    tf.app.run()
