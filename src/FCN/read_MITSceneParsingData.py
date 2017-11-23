import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob
import scipy.misc as misc
import tensorflow as tf
import TensorflowUtils as utils


def load_image(addr):
    return misc.imread(addr)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_dataset(data_dir):
    image_list = create_image_lists(data_dir)
    for k, record_list in image_list.items():
        tf_filename = k + '.tfrecords'
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(tf_filename)
        for record in record_list:
            # load the record
            image = load_image(record['image'])
            annotation = load_image(record['annotation'])
            filename = record['filename']
            label = record['label']

            # create a feature
            feature = {'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
                       'annotation': _bytes_feature(tf.compat.as_bytes(image.tostring())),
                       'filename': _bytes_feature(tf.compat.as_bytes((filename))),
                       'label': _int64_feature(label)}

            # create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()


def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {}
    file_glob_0 = os.path.join(image_dir, 'img0', '*.' + 'png')
    file_glob_0_list = glob.glob(file_glob_0)
    file_glob_1 = os.path.join(image_dir, 'img1', '*.' + 'png')
    file_glob_1_list = glob.glob(file_glob_1)

    for directory in directories:
        file_list = []
        image_list[directory] = []
        # file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')

        # print file_glob_0
        # print file_glob_0_list
        # file_list.extend(glob.glob(file_glob_0))
        # file_glob_1 = os.path.join(image_dir, directory,'1','*.' + 'png')
        # file_list.extend(glob.glob(file_glob_1))
        # file_list.extend(glob.glob(file_glob))
        if directory == 'training':
            start = 0
            end = 198
            # start = 0
            # end = 1000
        else:
            start = 0
            end = 198
            # start = 1000
            # end = 2000

        # file_glob_0 = os.path.join(image_dir,'raw-data', directory,'0','*.' + 'png')

        for f in file_glob_0_list[start:end]:

            #filename = os.path.splitext(f.split("/")[-1])[0]
            filename, _ = os.path.splitext(f)
            filename = os.path.basename(filename)
            annotation_file = os.path.join(image_dir, 'im_annotation', '0', filename + '-annotation.png')
            if os.path.exists(annotation_file):
                record = {'image': f, 'annotation': annotation_file, 'filename': filename, 'label': 0}
                image_list[directory].append(record)
            else:
                print("Annotation file not found for %s - Skipping" % filename)

        # file_glob_1 = os.path.join(image_dir, 'raw-data',directory,'1','*.' + 'png')

        for f in file_glob_1_list[start:end]:
            #filename = os.path.splitext(f.split("/")[-1])[0]
            filename, _ = os.path.splitext(f)
            filename = os.path.basename(filename)
            # annotation_file = os.path.join(image_dir, "15_15_annotation_new", "1", filename + '-annotation.png')
            annotation_file = os.path.join(image_dir, 'im_annotation', '1', filename + '-annotation.png')
            if os.path.exists(annotation_file):
                record = {'image': f, 'annotation': annotation_file, 'filename': filename, 'label': 1}
                image_list[directory].append(record)
            else:
                print("Annotation file not found for %s - Skipping" % filename)

        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list
