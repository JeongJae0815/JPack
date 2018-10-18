# -*- coding: utf-8 -*-
def load_tfrecord(self,)











def _int64_list_feature(values):
  """Returns a TF-Feature of int64_list.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, collections.Iterable):
    values = [values]

  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  def norm2bytes(value):
    return value.encode() if isinstance(value, str) and six.PY3 else value

  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))



def image_seg_to_tfexample(image_data, filename,height,width,label):

    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded':_bytes_list_feature(image_data),
        'image/filename':_bytes_list_feature(filename),
        'image/format':_bytes_list_feature(
            _IMAGE_FORMAT_MAP[FLAGS.image_format]),
        'image/height' : _int64_list_feature(height),
        'image/width'  : _int64_list_feature(width),
        'image/label'  : _int64_list_feature(label)
    }))
