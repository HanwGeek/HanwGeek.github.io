import os
import tensorflow as tf 

IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_data(filename_queue):
	# Reads and parses examples from CIFAR10 data files.
	class CIFAR10Record(object):
		pass

	result = CIFAR10Record()
	result.height = 32
	result.width = 32
	result.depth = 3

	lable_bytes = 1
	image_bytes = result.height * result.width * result.depth
	record_bytes = lable_bytes + image_bytes

	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	result.key, val = reader.read(filename_queue)

	# Convert data from string to uint8
	record_bytes = tf.decode_raw(val, tf.uint8)

	# Convert the first bytes of label from uint8 to int32
	# tf.strided_slice(data, begin, end)
	result.label = tf.cast(tf.strided_slice(
		record_bytes, [0], [lable_bytes]), tf.int32)

	# Reshape the image data from [depth * height * width]
	# to [depth, height, width]
	depth_major = tf.reshape(tf.strided_slice(
		record_bytes, [lable_bytes], 
		[lable_bytes + image_bytes]),
		[result.depth, result.height, result.width])

	# Convert image data from [depth, height, width] to
	# [height, width, depth]
	# tf.transpose(input, [d1, d2, d3, ...])
	result.uint8image = tf.transpose(depth_major, [1, 2, 0])

	return result

def generate_image_and_label_batch(image, label, 
	min_queue_examples, batch_size, shuffle):
	# Construct a queued batch of images and labels

	num_preprocess_threads = 8
	if shuffle:
		images, label_batch = tf.train.shuffle_batch(
			[image, label], batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size,
			min_after_dequeue=min_queue_examples)
	else:
		images, label_batch = tf.train.batch(
			[image, label], batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples+ 3 * batch_size)

	# Display the training images in the visualizer
	tf.summary.image('image', images)

	return images, tf.reshape(label_batch, [batch_size])

def inputs(eval_data, data_dir, batch_size):
	# Construct input for CIFAR evaluation using the Reader ops.

	if not eval_data:
		filenames = [os.path.join(
			data_dir, 'data_batch_%d.bin' % i)
				for i in range(1, 6)]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	else:
		filenames = [os.path.join(data_dir, 'test_batch.bin')]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

	# Create a filename queue to read
	filename_queue = tf.train.string_input_producer(filenames)

	# Read data
	read_input = read_data(filename_queue)
	reshaped_image = tf.cast(read_input.uint8image, 
		tf.float32)

	# Crop the central [height, width] of the image
	height = IMAGE_SIZE
	width = IMAGE_SIZE
	resized_image = tf.image.resize_image_with_crop_or_pad(
		reshaped_image, width, height)

	# Subtract off the mean and divide by the variance of the pixels
	float_image = tf.image.per_image_standardlization(resized_image)

	# Set the shape of tensors
	float_image.set_shape([height, width, 3])
	read_input.label.set_shape([1])

	# Ensure that the random shuffling has good mixing properties
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(
		num_examples_per_epoch * min_fraction_of_examples_in_queue)

	# Generate a batch
	return generate_image_and_label_batch(
		float_image,
		read_input.label,
		min_queue_examples,
		batch_size,
		shuffle=False)

def distorted_inputs(data_dir, batch_size):
	filenames = [os.path.join(data_dir, 'test_batch.bin')]
	num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

	# Create a filename queue to read
	filename_queue = tf.train.string_input_producer(filenames)

	# Read data
	read_input = read_data(filename_queue)
	reshaped_image = tf.cast(read_input.uint8image, 
		tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	# Randomly crop a [height, width] section of the image
	distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

	# Randomly filp the image horizontally
	distorted_image = tf.image.random_flip_left_right(distorted_image)

	# Randomly change the brightess and contrast
	distorted_image = tf.image.random_brightness(distorted_image,
		max_delta=63)
	distorted_image = tf.image.random_contrast(
		distorted_image, lower=0.2, upper=0.8)

	# Subtract off the mean and divide by the var of pix
	float_image = tf.image.per_image_standardization(distorted_image)

	# Set the shape of tensors
	float_image.set_shape([height, width, 3])
	read_input.label.set_shape([1])

	# Ensure that the random shuffling has good mixing properties
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(
		num_examples_per_epoch * min_fraction_of_examples_in_queue)

	# Generate a batch
	return generate_image_and_label_batch(
		float_image,
		read_input.label,
		min_queue_examples,
		batch_size,
		shuffle=True)	