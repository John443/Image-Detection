from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import math

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange
import tensorflow.contrib.slim as slim

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", 32, "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_string("checkpoint_dir", "checkpoint_dir/", "model to restore and save")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
tf.flags.DEFINE_integer('data_number', "30000", "number of data to evaluate")
tf.flags.DEFINE_string('subset', "validation", "validation of train, subset to evaluate")


# MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_SIZE = 224


# def vgg_net(weights, image):
# 	layers = (
# 		'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
#
# 		'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
#
# 		'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
# 		'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
#
# 		'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
# 		'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
#
# 		'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
# 		'relu5_3', 'conv5_4', 'relu5_4'
# 	)
#
# 	net = {}
# 	current = image
# 	for i, name in enumerate(layers):
# 		kind = name[:4]
# 		if kind == 'conv':
# 			kernels, bias = weights[i][0][0][0][0]
# 			# matconvnet: weights are [width, height, in_channels, out_channels]
# 			# tensorflow: weights are [height, width, in_channels, out_channels]
# 			kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
# 			bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
# 			current = utils.conv2d_basic(current, kernels, bias)
# 		elif kind == 'relu':
# 			current = tf.nn.relu(current, name=name)
# 			if FLAGS.debug:
# 				utils.add_activation_summary(current)
# 		elif kind == 'pool':
# 			current = utils.avg_pool_2x2(current)
# 		net[name] = current
#
# 	return net


def vgg_19(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           fc_conv_padding='VALID'):
	"""Oxford Net VGG 19-Layers version E Example.
		Note: All the fully_connected layers have been transformed to conv2d layers.
		To use in classification mode, resize input to 224x224.
		Args:
			inputs: a tensor of size [batch_size, height, width, channels].
			num_classes: number of predicted classes.
			is_training: whether or not the model is being trained.
			dropout_keep_prob: the probability that activations are kept in the dropout
				layers during training.
			spatial_squeeze: whether or not should squeeze the spatial dimensions of the
				outputs. Useful to remove unnecessary dimensions for classification.
			scope: Optional scope for the variables.
			fc_conv_padding: the type of padding to use for the fully connected layer
				that is implemented as a convolutional layer. Use 'SAME' padding if you
				are applying the network in a fully convolutional manner and want to
				get a prediction map downsampled by a factor of 32 as an output.
				Otherwise, the output prediction map will be (input / 32) - 6 in case of
				'VALID' padding.
		Returns:
			the last op containing the log predictions and end_points dict.
	"""
	with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
		end_points_collection = sc.name + '_end_points'
		# Collect outputs for conv2d, fully_connected and max_pool2d.
		with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
		                    outputs_collections=end_points_collection):
			net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
			net = slim.max_pool2d(net, [2, 2], scope='pool1')
			net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
			net = slim.max_pool2d(net, [2, 2], scope='pool2')
			net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
			net = slim.max_pool2d(net, [2, 2], scope='pool3')
			net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
			net = slim.max_pool2d(net, [2, 2], scope='pool4')
			net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
			net = slim.max_pool2d(net, [2, 2], scope='pool5')
			# Use conv2d instead of fully_connected layers.
			net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
			net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
			                   scope='dropout6')
			net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
			net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
			                   scope='dropout7')
			net = slim.conv2d(net, num_classes, [1, 1],
			                  activation_fn=None,
			                  normalizer_fn=None,
			                  scope='fc8')
			# Convert end_points_collection into a end_point dict.
			end_points = slim.utils.convert_collection_to_dict(end_points_collection)
			if spatial_squeeze:
				net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
				end_points[sc.name + '/fc8'] = net
			return net, end_points


def inference_vgg(image, keep_prob):
	"""
		Semantic segmentation network definition
		:param image: input image. Should have values in range 0-255
		:param keep_prob:
		:return:
		"""
	print("setting up vgg ...")

	pred, _ = vgg_19(image, num_classes=NUM_OF_CLASSESS, dropout_keep_prob=keep_prob, is_training=True)

	return pred


def inference(image, keep_prob):
	"""
	Semantic segmentation network definition
	:param image: input image. Should have values in range 0-255
	:param keep_prob:
	:return:
	"""
	print("setting up vgg initialized conv layers ...")
	# model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)
	#
	# mean = model_data['normalization'][0][0][0]
	# mean_pixel = np.mean(mean, axis=(0, 1))
	#
	# weights = np.squeeze(model_data['layers'])
	#
	# processed_image = utils.process_image(image, mean_pixel)

	with tf.variable_scope("inference"):
		# image_net = vgg_net(weights, processed_image)
		net, image_net = vgg_19(image,
		                        num_classes=NUM_OF_CLASSESS,
		                        dropout_keep_prob=keep_prob,
		                        spatial_squeeze=False,
		                        is_training=True)
		with tf.variable_scope('vgg_19') as sc:
			with tf.variable_scope('conv5') as layer:
				conv_final_layer = image_net[layer.name + '/conv5_3']
		# conv_final_layer = image_net["conv5_3"]

		pool5 = utils.max_pool_2x2(conv_final_layer)

		W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
		b6 = utils.bias_variable([4096], name="b6")
		conv6 = utils.conv2d_basic(pool5, W6, b6)
		relu6 = tf.nn.relu(conv6, name="relu6")
		if FLAGS.debug:
			utils.add_activation_summary(relu6)
		relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

		W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
		b7 = utils.bias_variable([4096], name="b7")
		conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
		relu7 = tf.nn.relu(conv7, name="relu7")
		if FLAGS.debug:
			utils.add_activation_summary(relu7)
		relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

		W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
		b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
		conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
		# annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

		# now to upscale to actual image size
		with tf.variable_scope('vgg_19') as sc:
			pool_4 = image_net[sc.name + "/pool4"]
			deconv_shape1 = pool_4.get_shape()
		# pool_4 = image_net['pool4']
		# deconv_shape1 = pool_4.get_shape()
		W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
		b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
		conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(pool_4))
		fuse_1 = tf.add(conv_t1, pool_4, name="fuse_1")

		with tf.variable_scope('vgg_19') as sc:
			pool_3 = image_net[sc.name + "/pool3"]
			deconv_shape2 = pool_3.get_shape()
		# pool_3 = image_net['pool3']
		# deconv_shape2 = pool_3.get_shape()
		W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
		b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
		conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(pool_3))
		fuse_2 = tf.add(conv_t2, pool_3, name="fuse_2")

		shape = tf.shape(image)
		deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
		W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
		b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
		conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

		annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

	return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list):
	optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
	grads = optimizer.compute_gradients(loss_val, var_list=var_list)
	if FLAGS.debug:
		# print(len(var_list))
		for grad, var in grads:
			utils.add_gradient_summary(grad, var)
	return optimizer.apply_gradients(grads)


def main(argv=None):
	keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
	image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
	annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

	pred_annotation, logits = inference(image, keep_probability)
	tf.summary.image("input_image", image, max_outputs=2)
	tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
	tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)

	loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
																		  labels=tf.squeeze(annotation, squeeze_dims=[3]),
																		  name="entropy")))
	tf.summary.scalar("entropy", loss)

	trainable_var = tf.trainable_variables()
	if FLAGS.debug:
		for var in trainable_var:
			utils.add_to_regularization_and_summary(var)
	train_op = train(loss, trainable_var)

	print("Setting up summary op...")
	summary_op = tf.summary.merge_all()

	print("Setting up image reader...")
	train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
	print(len(train_records))
	print(len(valid_records))

	print("Setting up dataset reader")
	image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
	if FLAGS.mode == 'train':
		train_dataset_reader = dataset.BatchDatset(train_records, image_options)
	validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

	config = tf.ConfigProto(allow_soft_placement=True,
	                        log_device_placement=False)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)

	print("Setting up Saver...")
	saver = tf.train.Saver()
	summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

	sess.run(tf.global_variables_initializer())
	ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print("Model restored...")

	if FLAGS.mode == "train":
		for itr in xrange(MAX_ITERATION):
			train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
			feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

			sess.run(train_op, feed_dict=feed_dict)

			if itr % 10 == 0:
				train_loss, summary_str, train_pred, logits_eval = sess.run(
					[loss, summary_op, pred_annotation, logits], feed_dict=feed_dict)

				train_annotations = np.squeeze(train_annotations, axis=3)
				train_pred = np.squeeze(train_pred, axis=3)
				sum_annotation = 0
				sum_pred_annotation = 0
				for index in range(FLAGS.batch_size):
					sum_annotation += np.sum(train_annotations[index])

					rows, cols = np.shape(train_pred[index])
					for i in range(rows):
						for j in range(cols):
							if train_pred[index][i, j] == 1 and train_annotations[index][i, j] == 1:
								sum_pred_annotation += 1

				acc = float(sum_pred_annotation) / sum_annotation

				print("Step: %d, Train_loss: %g, ACC: %f" % (itr, train_loss, acc))
				with open(os.path.join(FLAGS.logs_dir, 'train_log.txt'), 'a') as f:
					f.write("Step: %d, Train_loss: %g, ACC: %f" % (itr, train_loss, acc) + '\n')
				summary_writer.add_summary(summary_str, itr)

			if itr % 500 == 0:
				valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
				valid_loss, valid_pred = sess.run([loss, pred_annotation],
				                                  feed_dict={image: valid_images, annotation: valid_annotations,
				                                             keep_probability: 1.0})
				valid_annotations = np.squeeze(valid_annotations, axis=3)
				valid_pred = np.squeeze(valid_pred, axis=3)

				sum_pred = 0
				sum_annotation = 0
				sum_pred_annotation = 0

				for index in range(FLAGS.batch_size):
					sum_pred += np.sum(valid_pred[index])
					sum_annotation += np.sum(valid_annotations[index])

					rows, cols = np.shape(valid_pred[index])
					for i in range(rows):
						for j in range(cols):
							if valid_pred[index][i, j] == 1 and valid_annotations[index][i, j] == 1:
								sum_pred_annotation += 1
				acc = float(sum_pred_annotation) / sum_annotation
				with open(os.path.join(FLAGS.logs_dir, 'train_log.txt'), 'a') as f:
					f.write(
						"%s ---> Validation_loss: %g     acc: %f" % (datetime.datetime.now(), valid_loss, acc) + '\n')
				print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
				saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

	elif FLAGS.mode == "test":
		if FLAGS.subset == "train":
			train_dataset_reader = dataset.BatchDatset(train_records, image_options)

		num_iter = int(math.ceil(float(FLAGS.data_number) / FLAGS.batch_size))

		total_sum_annotation = 0
		total_sum_pred_annotation = 0

		for number in range(num_iter):

			sum_annotation = 0
			sum_pred_annotation = 0

			if FLAGS.subset == "validation":
				images_to_check, annotation_to_check = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
			elif FLAGS.subset == "train":
				images_to_check, annotation_to_check = train_dataset_reader.next_batch(FLAGS.batch_size)
			pred = sess.run(pred_annotation, feed_dict={image: images_to_check, annotation: annotation_to_check,
			                                            keep_probability: 1.0})
			annotation_to_check = np.squeeze(annotation_to_check, axis=3)
			pred = np.squeeze(pred, axis=3)

			for index in range(FLAGS.batch_size):
				sum_annotation += np.sum(pred[index])
				total_sum_annotation += np.sum(pred[index])
				rows, cols = np.shape(pred[index])
				for i in range(rows):
					for j in range(cols):
						if annotation_to_check[index][i, j] == 1 and pred[index][i, j] == 1:
							sum_pred_annotation += 1
							total_sum_pred_annotation += 1
			acc = float(sum_pred_annotation) / sum_annotation

			print("step:   " + str(number) + "               accuracy:    " + str(acc))

			# choose how many picture to show
			if number >= 0:
				for itr in range(FLAGS.batch_size):
					utils.save_image(images_to_check[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(number))
					utils.save_image(annotation_to_check[itr].astype(np.uint8) * 255.0, FLAGS.logs_dir,
					                 name="gt_" + str(number))
					utils.save_image(pred[itr].astype(np.uint8) * 255.0, FLAGS.logs_dir, name="pred_" + str(number))
					print("Saved image: %d" % number)

		total_acc = float(total_sum_pred_annotation) / total_sum_annotation

		print("total_acc:         " + str(total_acc))
		with open(os.path.join(FLAGS.logs_dir, 'eval_log.txt'), 'a') as f:
			f.write("number_data:       " + str(FLAGS.data_number) + '\n')
			f.write("test on:           " + str(FLAGS.subset) + '\n')
			f.write("total_acc:         " + str(total_acc) + '\n')


if __name__ == "__main__":
	tf.app.run()
