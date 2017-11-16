import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
# DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'


def read_dataset(data_dir):
	pickle_filename = "vgg_model_test.pickle"
	pickle_filepath = os.path.join(data_dir, pickle_filename)

	# utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
	# SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
	# result = create_image_lists(os.path.join(data_dir, SceneParsing_folder))
	result = create_image_lists(data_dir)
	print ("Pickling ...")
	with open(pickle_filepath, 'wb') as f:
		pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

	with open(pickle_filepath, 'rb') as f:
		result = pickle.load(f)
		training_records = result['training']
		validation_records = result['validation']
		del result

	return training_records, validation_records


def create_image_lists(image_dir):
	if not gfile.Exists(image_dir):
		print("Image directory '" + image_dir + "' not found.")
		return None
	directories = ['training', 'validation']
	image_list = {}
	file_glob_0 = os.path.join(image_dir, 'images', '0', '*.' + 'png')
	file_glob_0_list = glob.glob(file_glob_0)
	file_glob_1 = os.path.join(image_dir, 'images', '1', '*.' + 'png')
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
			end = -30000
			# start = 0
			# end = 1000
		else:
			start = -30000
			end = -1
			# start = 1000
			# end = 2000

		for f in file_glob_0_list[start:end]:

			filename = os.path.splitext(f.split("/")[-1])[0]
			annotation_file = os.path.join(image_dir, 'annotation', '0', filename + '-annotation.png')
			if os.path.exists(annotation_file):
				record = {'image': f, 'annotation': annotation_file, 'filename': filename, 'label': 0}
				image_list[directory].append(record)
			else:
				print("Annotation file not found for %s - Skipping" % filename)

		for f in file_glob_1_list[start:end]:
			filename = os.path.splitext(f.split("/")[-1])[0]

			annotation_file = os.path.join(image_dir, 'annotation', '1', filename + '-annotation.png')
			if os.path.exists(annotation_file):
				record = {'image': f, 'annotation': annotation_file, 'filename': filename, 'label': 1}
				image_list[directory].append(record)
			else:
				print("Annotation file not found for %s - Skipping" % filename)

		random.shuffle(image_list[directory])
		no_of_images = len(image_list[directory])
		print ('No. of %s files: %d' % (directory, no_of_images))

	return image_list
