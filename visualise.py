#!/usr/bin/env python3
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

image_directory = 'train/images/'
annotation_file = 'train/cracks.json'
example_coco = COCO(annotation_file)

categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

category_names = set([category['supercategory'] for category in categories])
print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))

category_ids = example_coco.getCatIds(catNms=['crack'])
image_ids = example_coco.getImgIds(catIds=category_ids)

for i in image_ids: 
	image_data = example_coco.loadImgs(image_ids[i])[0]
	print(image_data)
	image = io.imread(image_directory + image_data['file_name'])
	print(image_directory + image_data['file_name'])
	plt.imshow(image); plt.axis('off')
	pylab.rcParams['figure.figsize'] = (8.0, 10.0)
	annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
	annotations = example_coco.loadAnns(annotation_ids)
	example_coco.showAnns(annotations)
	plt.savefig('seg_images/'+image_data['file_name'],dpi=600)
	plt.show()
	plt.clf()
	

