# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import dataset_utils


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer',
}
class TFRecordDataset:
    def __init__(self, tfrecord_dir, dataset_name, num_classes):
        self.tfrecord_dir = tfrecord_dir
        self.dataset_name = dataset_name
        self.num_classes = num_classes

    def __get_num_samples__(self, split_name):
        # Count the total number of examples in all of these shard
        num_samples = 0
        file_pattern_for_counting = split_name
        tfrecords_to_count = [os.path.join(self.tfrecord_dir, file) for file in os.listdir(self.tfrecord_dir) if
                              file.startswith(file_pattern_for_counting)]
        for tfrecord_file in tfrecords_to_count:
            for record in tf.python_io.tf_record_iterator(tfrecord_file):
                num_samples += 1

        return num_samples

    def get_split(self, split_name):
        splits_to_sizes = self.__get_num_samples__(split_name)

        file_pattern = split_name + '-*.tfrecord'
        file_pattern = os.path.join(self.tfrecord_dir, file_pattern)
        reader = tf.TFRecordReader

        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/class/label': tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        }

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)
        labels_to_names = None
        if dataset_utils.has_labels(self.tfrecord_dir):
            labels_to_names = dataset_utils.read_label_file(self.tfrecord_dir)
        #print dataset_utils.has_labels(self.tfrecord_dir)
        return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=splits_to_sizes,
            items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
            num_classes=self.num_classes,
            labels_to_names=labels_to_names)


def load_batch(dataset, batch_size=15, height=224, width=224, num_classes=17, is_training=True):
    """Loads a single batch of data.

    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.

    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    # Creates a TF-Slim DataProvider which reads the dataset in the background during both training and testing.
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    [image, label] = provider.get(['image', 'label'])

    # image: resize with crop
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    image = tf.to_float(image)

    # label: one-hot encoding
    one_hot_labels = slim.one_hot_encoding(label, num_classes)

    # Batch it up.
    images, labels = tf.train.batch(
        [image, one_hot_labels],
        batch_size=batch_size,
        num_threads=2,
        capacity=2 * batch_size)

    return images, labels, dataset.num_samples
