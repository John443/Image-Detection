"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import FCN
import tensorflow as tf


class BatchDatset:
    data_path = ""
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, data_path, batch_size, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.batch_size = batch_size
        self.data_path = data_path
        self.image_options = image_options
        self.graph, self.images, self.annotations, self.random_images, self.random_annotations = self._build_graph()
        #start a session
        with tf.Session(graph=self.graph) as self.sess:
            self.sess.run(tf.local_variables_initializer())

            # Create a coordinator and run all QueueRunner objects
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(coord=self.coord)

    def __del__(self):
        # Stop the threads
        self.coord.request_stop()

        # Wait for threads to stop
        self.coord.join(self.threads)
        self.sess.close()

    def _build_graph(self):
        self.feature = {'image': tf.FixedLenFeature([], tf.string),
                        'annotation': tf.FixedLenFeature([], tf.string),
                        'filename': tf.VarLenFeature([], tf.string),
                        'label': tf.FixedLenFeature([], tf.int64)}
        g = tf.Graph()
        with g.as_default():

            # Create a list of filenames and pass it to a queue
            filename_queue = tf.train.string_input_producer([self.data_path], num_epochs=10000, shuffle=True)

            # Define a reader and read the next record
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            # Decode the record read by the reader
            features = tf.parse_single_example(serialized_example, features=self.feature)

            # Convert the image data from string back to the numbers
            image = tf.decode_raw(features['image'], tf.uint8)
            annotation = tf.decode_raw(features['annotation'], tf.uint8)
            filename = features['filename']
            label = tf.cast(features['label'], tf.int32)

            # Reshape image data into the original shape
            image = tf.reshape(image, [FCN.IMAGE_SIZE, FCN.IMAGE_SIZE, 3])
            annotation = tf.reshape(image, [FCN.IMAGE_SIZE * FCN.IMAGE_SIZE, ])

            # preprocess
            self.__channels = True
            image = np.array([self._transform(image) for filename in self.files])
            self.__channels = False
            annotation = np.array([np.expand_dims(self._transform(annotation), axis=3) for filename in self.files])

            # Creates batches
            images, annotations, _, _ = tf.train.batch([image, annotation, filename, label],
                                                batch_size=self.batch_size,
                                                capacity=3 * self.batch_size,
                                                num_threads=1)

            # create random batches
            random_images, random_annotations, _, _ = tf.train.shuffle_batch([image, annotation, filename, label],
                                                                            batch_size= self.batch_size,
                                                                            capacity=3 * self.batch_size,
                                                                            num_threads=1,
                                                                            min_after_dequeue= self.batch_size)

        return g, images, annotations, random_images, random_annotations

    def _transform(self, image):
        # image = misc.imread(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        '''
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]
        '''
        return self.sess.run([self.images, self.annotations])

    def get_random_batch(self, batch_size):
        '''
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
        '''
        return self.sess.run([self.random_images, self.random_annotations])
