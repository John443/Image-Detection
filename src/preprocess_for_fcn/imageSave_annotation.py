import re
from scipy import misc
import scipy as sp
import numpy as np
import cv2
import scipy.ndimage
import os
import random
import tensorflow as tf

# a = np.random.random((500,500,3))
# misc.imsave('test.png',a)
# b = misc.imread('test.png')
# print type(b)
# print b
# print "done"

CENTER_SIZE = 128
TOTAL_SIZE = 224
STRIDE = CENTER_SIZE - 1
CANCER_COLOR = 1.0
NO_CANCER_COLOR = 0.0
if (TOTAL_SIZE - CENTER_SIZE) % 2 != 0:
	ITEM_ADD = (TOTAL_SIZE - CENTER_SIZE + 1) / 2
else:
	ITEM_ADD = (TOTAL_SIZE - CENTER_SIZE) / 2


def flood_fill(test_array, h_max=CANCER_COLOR):

	input_array = np.copy(test_array) 
	el = sp.ndimage.generate_binary_structure(2, 2).astype(np.int)
	inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
	output_array = np.copy(input_array)
	output_array[inside_mask] = h_max
	output_old_array = np.copy(input_array)
	output_old_array.fill(0)   
	el = sp.ndimage.generate_binary_structure(2, 1).astype(np.int)
	while not np.array_equal(output_old_array, output_array):
		output_old_array = np.copy(output_array)
		output_array = np.maximum(input_array, sp.ndimage.grey_erosion(output_array, size=(3, 3), footprint=el))
	return output_array


def perturb(image, annotation, index, dir_name, filename_image, dir_label_annotation, crop_image_name_annotation):
	img = tf.image.random_brightness(image, max_delta=64 / 255)
	img = tf.image.random_saturation(img, lower=0, upper=0.25)
	img = tf.image.random_hue(img, max_delta=0.04)
	img = tf.image.random_contrast(img, lower=0, upper=0.75)
	config = tf.ConfigProto(allow_soft_placement=True,
							log_device_placement=False)
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		img_val = sess.run(img)
	cv2.imwrite(dir_name + filename_image[:-4] + '-' + str(index) + '-ptb.png', img_val)
	annotation[annotation == 1] = 255
	cv2.imwrite(dir_label_annotation + crop_image_name_annotation[:-15] + '-' + str(index) + '-ptb-annotation.png',
				annotation)


def augmentation(crop_image, dir_name, filename_image, crop_image_annotation, crop_image_name_annotation,
				 dir_label_annotation):

	# center = (CENTER_SIZE / 2 - 0.5, CENTER_SIZE / 2 - 0.5)
	images = []
	annotations = []
	center = (TOTAL_SIZE / 2, TOTAL_SIZE / 2)
	M1 = cv2.getRotationMatrix2D(center, 90, 1)

	image1 = cv2.warpAffine(crop_image, M1, (TOTAL_SIZE, TOTAL_SIZE))
	annotation1 = cv2.warpAffine(crop_image_annotation, M1, (TOTAL_SIZE, TOTAL_SIZE))
	images.append(image1)
	annotations.append(annotation1)

	image2 = cv2.flip(image1, 1)
	annotation2 = cv2.flip(annotation1, 1)
	images.append(image2)
	annotations.append(annotation2)

	image3 = cv2.warpAffine(image1, M1, (TOTAL_SIZE, TOTAL_SIZE))
	annotation3 = cv2.warpAffine(annotation1, M1, (TOTAL_SIZE, TOTAL_SIZE))
	images.append(image3)
	annotations.append(annotation3)

	image4 = cv2.flip(image3, 1)
	annotation4 = cv2.flip(annotation3, 1)
	images.append(image4)
	annotations.append(annotation4)

	image5 = cv2.warpAffine(image3, M1, (TOTAL_SIZE, TOTAL_SIZE))
	annotation5 = cv2.warpAffine(annotation3, M1, (TOTAL_SIZE, TOTAL_SIZE))
	images.append(image5)
	annotations.append(annotation5)

	image6 = cv2.flip(image5, 1)
	annotation6 = cv2.flip(annotation5, 1)
	images.append(image6)
	annotations.append(annotation6)

	image7 = cv2.warpAffine(image5, M1, (TOTAL_SIZE, TOTAL_SIZE))
	annotation7 = cv2.warpAffine(annotation5, M1, (TOTAL_SIZE, TOTAL_SIZE))
	images.append(image7)
	annotations.append(annotation7)

	image8 = cv2.flip(image7, 1)
	annotation8 = cv2.flip(annotation7, 1)
	images.append(image8)
	annotations.append(annotation8)

	for i in range(len(images)):
		cv2.imwrite(dir_name + filename_image[:-4] + '-' + str(i + 1) + '.png', images[i])
		annot = annotations[i].copy()
		annot[annot == 1] = 255
		cv2.imwrite(dir_label_annotation + crop_image_name_annotation[:-15] + '-' + str(i + 1) + '-annotation.png', annot)
		# perturb(images[i], annotations[i], i + 1, dir_name, filename_image, dir_label_annotation, crop_image_name_annotation)


def preprocess(filename_svg, filename_image, dir_name, base_name):

	s = ''
	with open(filename_svg) as f:
		for line in f.readlines():
			s += line

	pattern = 'points=(.*?)stroke'
	pattern = re.compile(pattern)
	results = pattern.findall(s)

	canvas_size = 2048 + TOTAL_SIZE - CENTER_SIZE

	canvas = np.zeros((2048, 2048), dtype='uint8')
	canvas.fill(NO_CANCER_COLOR)

	center_example = np.zeros((CENTER_SIZE, CENTER_SIZE))
	center_example.fill(NO_CANCER_COLOR)
	sum_shelter = np.sum(center_example)

	# points = []

	for result in results:
		result = result[1:-2].split(' ')
		for i in range(len(result)-1):
			current_points = [int(item) for item in result[i].split(',')]
			next_points = [int(item) for item in result[i+1].split(',')]
			cv2.line(canvas, (current_points[0], current_points[1]), (next_points[0], next_points[1]), CANCER_COLOR, 3)

	M = np.float32([[1, 0, ITEM_ADD], [0, 1, ITEM_ADD]])
	canvas = cv2.warpAffine(canvas, M, (canvas_size, canvas_size))

	# cv2.imwrite('opencv_line.png', canvas)

	canvas = flood_fill(canvas)

	# misc.imsave('fill.png', canvas)

	image = cv2.imread(filename_image, 1)

	image = cv2.warpAffine(image, M, (canvas_size, canvas_size))
	# cv2.imwrite('image_moved.png',image)

	dir_name_label_1 = dir_name + '1' + '/'
	dir_name_label_0 = dir_name + '0' + '/'

	dir_annotation_1 = dir_name[:-1]+'_annotation/' + '1' + '/'
	dir_annotation_0 = dir_name[:-1]+'_annotation/' + '0' + '/'

	if not os.path.exists(dir_name):
		os.mkdir(dir_name)
	if not os.path.exists(dir_name[:-1]+'_annotation/'):
		os.mkdir(dir_name[:-1]+'_annotation/')
	if not os.path.exists(dir_name_label_0):
		os.mkdir(dir_name_label_0)
	if not os.path.exists(dir_name_label_1):
		os.mkdir(dir_name_label_1)
	if not os.path.exists(dir_annotation_1):
		os.mkdir(dir_annotation_1)
	if not os.path.exists(dir_annotation_0):
		os.mkdir(dir_annotation_0)

	row_number = 0
	row_start_index = 0
	while row_start_index + TOTAL_SIZE < canvas_size:
		col_start_index = 0
		while col_start_index + TOTAL_SIZE < canvas_size:
			single_crop_image(image, row_start_index, col_start_index, canvas, dir_name_label_0, dir_annotation_0,
							  base_name, 1, sum_shelter, dir_name_label_1, dir_annotation_1)
			col_start_index += STRIDE

		col_start_index = canvas_size - TOTAL_SIZE
		single_crop_image(image, row_start_index, col_start_index, canvas, dir_name_label_0, dir_annotation_0,
						  base_name, 1, sum_shelter, dir_name_label_1, dir_annotation_1)
		row_start_index += STRIDE
		row_number += 1

	row_start_index = canvas_size - TOTAL_SIZE
	col_start_index = 0
	while col_start_index + TOTAL_SIZE < canvas_size:
		single_crop_image(image, row_start_index, col_start_index, canvas, dir_name_label_0, dir_annotation_0,
						  base_name, 1, sum_shelter, dir_name_label_1, dir_annotation_1)
		col_start_index += STRIDE

	col_start_index = canvas_size - TOTAL_SIZE
	single_crop_image(image, row_start_index, col_start_index, canvas, dir_name_label_0, dir_annotation_0,
					  base_name, 1, sum_shelter, dir_name_label_1, dir_annotation_1)
	row_start_index += STRIDE

	print(str(base_name)+"                done")


def add_jitter(index):
	offset = random.randint(0, 8)

	canvas_size = 2048 + TOTAL_SIZE - CENTER_SIZE
	if index + TOTAL_SIZE < canvas_size:
		new_start_index = index + offset
	else:
		new_start_index = index - offset

	return new_start_index


def single_crop_image(image, row_start_index, col_start_index, canvas, dir_name_label_0, dir_annotation_0,
					  base_name, c_type, sum_shelter=None, dir_name_label_1=None, dir_annotation_1=None):
	# c_type=1 for cancer image and c_type=2 for non_cancer
	# row_start_index = add_jitter(row_start_index)
	# col_start_index = add_jitter(col_start_index)

	crop_image = image[row_start_index:(row_start_index + TOTAL_SIZE),
					   col_start_index:(col_start_index + TOTAL_SIZE)]
	crop_image_annotation = canvas[row_start_index:(row_start_index + TOTAL_SIZE),
								   col_start_index:(col_start_index + TOTAL_SIZE)]
	if c_type == 2:
		dir_name_label = dir_name_label_0
		dir_label_annotation = dir_annotation_0
	else:
		if np.sum(canvas[(row_start_index + ITEM_ADD):(row_start_index + ITEM_ADD + CENTER_SIZE),
						 (col_start_index + ITEM_ADD):(col_start_index + ITEM_ADD + CENTER_SIZE)]) > sum_shelter:
			dir_name_label = dir_name_label_1
			dir_label_annotation = dir_annotation_1
		else:
			dir_name_label = dir_name_label_0
			dir_label_annotation = dir_annotation_0

	crop_image_name = str(base_name) + '-' + str(row_start_index) + '-' + str(col_start_index) + '.png'
	crop_image_name_annotation = str(base_name) + '-' + str(row_start_index) + '-' + str(col_start_index) \
												+ '-annotation' + '.png'
	cv2.imwrite(dir_name_label + crop_image_name, crop_image)
	crop_image_annotation[crop_image_name_annotation == 1] = 255
	cv2.imwrite(dir_label_annotation + crop_image_name_annotation, crop_image_annotation)
	if c_type == 1 and dir_name_label == dir_name_label_1:
		augmentation(crop_image, dir_name_label, crop_image_name, crop_image_annotation,
					 crop_image_name_annotation, dir_label_annotation)


def preprocess_non_cancer(filename_image, dir_name, base_name):
	canvas_size = 2048 + TOTAL_SIZE - CENTER_SIZE
	canvas = np.zeros((2048, 2048), dtype='uint8')
	canvas.fill(NO_CANCER_COLOR)
	M = np.float32([[1, 0, ITEM_ADD], [0, 1, ITEM_ADD]])
	canvas = cv2.warpAffine(canvas, M, (canvas_size, canvas_size))
	
	M = np.float32([[1, 0, ITEM_ADD], [0, 1, ITEM_ADD]])
	image = cv2.imread(filename_image, 1)
	image = cv2.warpAffine(image, M, (canvas_size, canvas_size))

	dir_name_label_0 = dir_name+'0'+'/'
	dir_annotation_0 = dir_name[:-1]+'_annotation/'+'0'+'/'

	if not os.path.exists(dir_name):
		os.mkdir(dir_name)
	if not os.path.exists(dir_name[:-1]+'_annotation/'):
		os.mkdir(dir_name[:-1]+'_annotation/')
	if not os.path.exists(dir_name_label_0):
		os.mkdir(dir_name_label_0)
	if not os.path.exists(dir_annotation_0):
		os.mkdir(dir_annotation_0)

	row_number = 0
	row_start_index = 0
	while row_start_index + TOTAL_SIZE < canvas_size:
		col_start_index = 0
		while col_start_index + TOTAL_SIZE < canvas_size:
			single_crop_image(image, row_start_index, col_start_index, canvas, dir_name_label_0, dir_annotation_0,
							  base_name, 2)
			col_start_index += STRIDE
		col_start_index = canvas_size - TOTAL_SIZE
		single_crop_image(image, row_start_index, col_start_index, canvas, dir_name_label_0, dir_annotation_0,
						  base_name, 2)
		row_start_index += STRIDE
		row_number += 1

	row_start_index = canvas_size - TOTAL_SIZE
	col_start_index = 0
	while col_start_index + TOTAL_SIZE < canvas_size:
		single_crop_image(image, row_start_index, col_start_index, canvas, dir_name_label_0, dir_annotation_0,
						  base_name, 2)
		col_start_index += STRIDE
	col_start_index = canvas_size - TOTAL_SIZE
	single_crop_image(image, row_start_index, col_start_index, canvas, dir_name_label_0, dir_annotation_0,
					  base_name, 2)
	row_start_index += STRIDE
	print(str(base_name)+"                done")


if __name__ == "__main__":
	filename_svg = '../data/label/2017-06-09_18.08.16.ndpi.16.14788_15256.2048x2048.svg'
	filename_image = '../data/cancer/2017-06-09_18.08.16.ndpi.16.14788_15256.2048x2048.tiff'
	dir_name = '../data/img'
	non_cancer_image = "non_cancer.tiff"
	preprocess(filename_svg, filename_image, dir_name, '2017-06-09_18.08.16.ndpi.16.14788_15256.2048x2048')
	preprocess_non_cancer(non_cancer_image, dir_name, 'qwertyui')