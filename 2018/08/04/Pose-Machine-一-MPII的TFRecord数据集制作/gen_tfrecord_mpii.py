import tensorflow as tf
import scipy.io as sio
import numpy as np
import operator 
import random
import math
import sys
import os

MPII_MAT_FILE = '../mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
IMG_DIR = '../images'


DATASET_DIR = './data/tfrecords'
SPLIT_PATH = './data/lists/images_mpii_{}.txt'

# Seed for repeatability
RANDOM_SEED = 42

# The number of shards per dataset split
NUM_SHARDS = 20

# The num of joints in pose
NUM_JOINTS = 16

class ImageReader():
	# Helper to create image for tensorflow

	def __init__(self):
		self._decode_jpeg_data = tf.placeholder(
			dtype=tf.string)
		self._decode_jpeg = tf.image.decode_jpeg(
			self._decode_jpeg_data, channels=3)

	def decode_jpeg(self, sess, image_data):
		image = sess.run(self._decode_jpeg,
			feed_dict={self._decode_jpeg_data: image_data})
		return image

	def read_image_dims(self, sess, image_data):
		image = sess.run(self._decode_jpeg,
			feed_dict={self._decode_jpeg_data: image_data})
		return image.shape[0], image.shape[1]


def int64_feature(val):
	if not isinstance(val, (tuple, list)):
		val = [val]
	return tf.train.Feature(
		int64_list=tf.train.Int64List(value=val))


def float_feature(val):
	return tf.train.Feature(
		float_list=tf.train.FloatList(value=val))


def bytes_feature(val):
	return tf.train.Feature(
		bytes_list=tf.train.BytesList(value=[val]))

def image_to_tfexample(image_data, image_format, 
	height, width, pose, label):
		return tf.train.Example(
			features=tf.train.Features(feature={
				'image/encoded': bytes_feature(image_data),
				'image/format': bytes_feature(image_format),
				'image/class/pose': int64_feature([int(el) for el in pose]),
				'image/class/label': int64_feature(label),
				'image/height': int64_feature(height),
				'image/width': int64_feature(width),
			}))

def get_action_class(act_name, act_list, act_id):
	try:
		if act_name not in act_list:
			act_list[act_name] = (len(act_list.keys()), set([act_id]))
		else:
			act_list[act_name][1].add(act_id)
			# A Dataset bug here
		return act_list[act_name][0]
	except Exception as e:
		print('Invalid class name {}. setting -1. {}'.format(act_name, e))
		return -1

def get_dataset_filename(dataset_dir, shard_id):
	filename = 'mpii_%05d-of-%05d.tfrecord' % (
		shard_id, NUM_SHARDS)
	return os.path.join(dataset_dir, filename)

def construct_dataset(lists_to_write, dataset_dir):
	num_per_shard = int(math.ceil(len(lists_to_write) / float(NUM_SHARDS)))

	with tf.Graph().as_default():
		image_reader = ImageReader()

		with tf.Session() as sess:
			for shard_id in range(NUM_SHARDS):
				filename = get_dataset_filename(dataset_dir,
					shard_id)

				with tf.python_io.TFRecordWriter(filename) as tfrecord_writer:
					start_idx = shard_id * num_per_shard
					end_idx = min((shard_id + 1) * num_per_shard, len(lists_to_write))
					for i in range(start_idx, end_idx):
						sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
							i + 1, len(lists_to_write), shard_id))
						sys.stdout.flush()

						# Read image and pose
						fname = os.path.join(IMG_DIR, lists_to_write[i][0])
						label = lists_to_write[i][1]
						poses = lists_to_write[i][2]
						all_joints = []
						for pose in poses:
							joints = dict((el[0], [el[1], el[2], el[3]]) for el in pose)
							final_pose = []
							for i in range(NUM_JOINTS):
								if i in joints:
									final_pose.append(joints[i])
								else:
									final_pose.append([-1, -1, 0])
							final_pose = [item for sublist in final_pose for item in sublist]
							all_joints += final_pose
						assert(len(all_joints) % 16 == 0)
						# if i == 1249:
						# 	print(all_joints)	
						image_data = tf.gfile.FastGFile(fname, 'rb').read()
						height, width = image_reader.read_image_dims(sess, image_data)

						example = image_to_tfexample(
							image_data, b'jpg', height, width, all_joints, label)
						tfrecord_writer.write(example.SerializeToString())

	sys.stdout.write('\n')
	sys.stdout.flush()


def main():
	# Read annotation mat
	T = sio.loadmat(MPII_MAT_FILE, squeeze_me=True,
			struct_as_record=False)
	M = T['RELEASE']
	annots = M.annolist
	is_train = M.img_train
	label = M.act

	# Define lists
	# splits = ['train', 'val', 'test']
	# lists_to_write = {}
	# img_id_in_split = {}
	# splits_filenames = {}
	# filename_to_split = {}
	# all_imnames = []
	# for spl in splits:
	# 	lists_to_write[spl] = []
	# 	img_id_in_split[spl] = []
	actclassname_to_id = {}
	lists_to_write = []
	img_id_list = []

	for aid, annot in enumerate(annots):
		img_name = annot.image.name
		points_fmted = []
		if 'annorect' in dir(annot):
			rects = annot.annorect
			if isinstance(rects, sio.matlab.mio5_params.mat_struct):
				rects = np.array([rects])
				for rect in rects:
					points_rect = []
					try:
						points = rect.annopoints.point
					except:
						continue
					for point in points:
						if point.is_visible in [0, 1]:
							is_visible = point.is_visible
						else:
							is_visible = 0
						points_rect.append((point.id, point.x, point.y, is_visible))
					points_fmted.append(points_rect)

		# Construct image and label obj
		image_obj = (annot.image.name,
			get_action_class(label[aid].act_name,
				actclassname_to_id,
				label[aid].act_id),
				points_fmted)
		if os.path.exists(os.path.join(IMG_DIR, img_name)):
			lists_to_write.append(image_obj)
			img_id_list.append(aid+1)

	# Get all act classes
	cls_ids = sorted(actclassname_to_id.items(), 
			key=operator.itemgetter(1))
	print('Total classes found: {}'.format(len(cls_ids)))

	# Write out act class names

	# Randomize the train set
	random.seed(RANDOM_SEED)
	lists_to_write = random.sample(lists_to_write, len(lists_to_write))
	construct_dataset(lists_to_write, DATASET_DIR)

	print('\nFinished converting dataset!')


if __name__ == '__main__':
	main()