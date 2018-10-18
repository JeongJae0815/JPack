from random import shuffle
import glob
import tensorflow as tf
import cv2
import sys
import numpy as np

#image read
from PIL import Image # image class read
import skimage.io as io # image rgb read

#data_list read

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.

    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image = mask = line.strip("\n")
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + '/JPEGImages/'+image+'.jpg')
        masks.append(data_dir + '/SegmentationClass/'+mask+'.png')
    return images, masks

def write_images_from_disk_tfrecord(img_list, label_list,tfrecords_filename): # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.

    Args:

    Returns:
      Two tensors: the decoded image and its mask.
    """
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    # Let's collect the real images to later on compare
    # to the reconstructed ones
    original_images = []
    filename_pairs=zip(img_list,label_list)
    i=0
    for img_path, annotation_path in filename_pairs:
        i=i+1
        if not i % 100:
            print('Train data: {}/{}'.format(i, len(img_list)))
            sys.stdout.flush()

        img = np.array(Image.open(img_path))
        annotation = np.array(Image.open(annotation_path))

        # The reason to store image sizes was demonstrated
        # in the previous example -- we have to know sizes
        # of images to later read raw serialized string,
        # convert to 1d array and convert to respective
        # shape that image used to have.
        height = img.shape[0]
        width = img.shape[1]

        # Put in the original images into array
        # Just for future check for correctness
        #original_images.append((img, annotation))

        img_raw = img.tostring()
        annotation_raw = annotation.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

class ImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list,tfrecords_filename,input_size):#, input_size, random_scale, random_mirror, img_mean, coord):
        '''Initialise an ImageReader.
        
        Args:
          train_data_dir: ////////////////////path to the directory with images and masks.
          label_data_dir: ////////////////////path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          ignore_label: index of label to ignore during the training.
          img_mean: vector of mean colour values.
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.tfrecords_filename = tfrecords_filename

        self.image_list, self.label_list = read_labeled_image_list(self.data_dir, self.data_list)               
        write_images_from_disk_tfrecord(self.image_list, self.label_list,self.tfrecords_filename) 
        

    
    def read_and_decode(self,tfrecords_filename,epochs,batch_size):
        filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=epochs)

        IMAGE_HEIGHT, IMAGE_WIDTH = self.input_size    
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string)
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        annotation = tf.decode_raw(features['mask_raw'], tf.uint8)

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)

        image_shape = tf.stack([height, width, 3])
        annotation_shape = tf.stack([height, width, 1])

        image = tf.reshape(image, image_shape)
        annotation = tf.reshape(annotation, annotation_shape)

        image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)
        annotation_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.int32)

        # Random transformations can be put here: right before you crop images
        # to predefined size. To get more information look at the stackoverflow
        # question linked above.

        resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                               target_height=IMAGE_HEIGHT,
                                               target_width=IMAGE_WIDTH)

        resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation,
                                               target_height=IMAGE_HEIGHT,
                                               target_width=IMAGE_WIDTH)


        images, annotations = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                                     batch_size=batch_size,
                                                     capacity=30,
                                                     num_threads=2,
                                                     min_after_dequeue=10)

        return images, annotations


