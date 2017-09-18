import glob 
import re
import multiprocessing
from imageSave_annotation import preprocess_non_cancer

dir_svg = 'labels'
dir_cancer = '/disk1/cancer'
dir_non_cancer = 'non_cancer'
non_cancers = glob.glob(dir_non_cancer+'/*.tiff')
pattern = dir_non_cancer+'/(.*)?.tiff'
pattern = re.compile(pattern)
dir_name = '/disk1/15_15_128_image/'
pool = multiprocessing.Pool()


for non_cancer in non_cancers:
    print non_cancer
    base = pattern.match(non_cancer).group(1)
    image_name = non_cancer
    pool.apply_async(preprocess_non_cancer, (image_name, dir_name, base, ))

pool.close()
pool.join()
print "done"
