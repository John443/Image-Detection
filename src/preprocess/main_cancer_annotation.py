import glob
import os
import re
import multiprocessing
from imageSave_annotation import preprocess

dir_svg = '../../data/labels'
dir_cancer = '../../data/cancer'
dir_non_cancer = 'non_cancer'
labels = glob.glob(dir_svg + '/*.svg')
# labels = os.listdir(dir_svg)
pattern = dir_svg + '/(.*)?.svg'
pattern = re.compile(pattern)
dir_name = '../../data/image/'
pool = multiprocessing.Pool()


for label_name in labels:
	print(label_name)
	if label_name < "2017-06-13_14.46.49.ndpi.16.27271_20873.2048x2048":
		continue
	base = pattern.match(label_name).group(1)
	image_name = dir_cancer + '/' + base + '.tiff'
	pool.apply_async(preprocess, (label_name, image_name, dir_name, base, ))

pool.close()
pool.join()
print("done")
