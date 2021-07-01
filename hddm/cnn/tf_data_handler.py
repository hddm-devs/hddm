import tensorflow as tf
import tqdm
import numpy as np
import os


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecord(
        tf_record_dir='',
        tf_filename='',
        images=None,
        labels=None):
    writer = tf.python_io.TFRecordWriter(os.path.join(
        tf_record_dir,
        tf_filename))
    for i in tqdm.tqdm(range(len(images))):
        feature = {'label': _bytes_feature(labels[i].tostring()),
                   'image': _bytes_feature(images[i].tostring())}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()

# establish the data queue
def inputs(
        tfrecord_file,
        num_epochs,
        target_data_dims,
        target_label_dims,
        batch_size):

    with tf.name_scope('input'):
        if os.path.exists(tfrecord_file) is False:
            print("{} not exists".format(tfrecord_file))
        feature = {
            'label': tf.FixedLenFeature([], tf.string),
            'image': tf.FixedLenFeature([], tf.string)
        }

        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([tfrecord_file], num_epochs=num_epochs)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['image'], tf.float32)
        label = tf.decode_raw(features['label'], tf.float32)

        # reshape the label into specific dimensions
        label = tf.reshape(label, np.asarray(target_label_dims[-2:]))
        # Reshape image data into the original shape
        image = tf.reshape(image, np.asarray(target_data_dims[-2:]))
        # Creates batches by randomly shuffling tensors
        params, hists = tf.train.batch([image, label], batch_size=batch_size, capacity=30,
                                                num_threads=2)
    return params, hists
