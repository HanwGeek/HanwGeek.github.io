import tensorflow as tf 
import data_input
import re
import os

IMAGE_SIZE = data_input.IMAGE_SIZE
NUM_CLASSES = data_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = data_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
BATCH_SIZE = 128
DATA_DIR = './cifar-10-batches-bin/'
TRAIN_DIR = './cifar10_train'

# Hyperparameters
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LERANING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

# name of the summaries when visualizing a model
TOWER_NAME = 'tower'

def _activation_summary(x):
	# Helper to create summaries for activations
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
	# Helper to create a Variable stored on CPU memory
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer=initializer)
	return var


def _variable_with_weight_decay(name, shape, stddev, wd):
	# Helper to create an initialized Variable with weight decay
	var = _variable_on_cpu(name, shape,
		tf.truncated_normal_initializer(stddev=stddev))

	# wd: add L2Loss weight decay multiplied by this float
	# tf.nn.l2_loss output = sum(t ** 2) / 2
	if wd:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var


def distorted_inputs():
	# Construct distorted input
	images, labels = data_input.distorted_inputs(
		data_dir=DATA_DIR, batch_size=BATCH_SIZE)
	return images, labels


def inputs(eval_data):
	# Construct input
	images, labels = data_input.inputs(
		eval_data=eval_data, data_dir=DATA_DIR,
		batch_size=BATCH_SIZE)
	return images, labels


def inference(images):
	# Build the model AlexNet
	# conv1
	with tf.variable_scope('conv1') as scope:
		kernel = _variable_with_weight_decay('weights',
			shape=[5, 5, 3, 32], stddev=1e-4, wd=0.0)
		conv = tf.nn.conv2d(images, kernel, 
			[1, 1, 1, 1], padding="SAME")
		biases = _variable_on_cpu('biases', [32], 
			tf.constant_initializer(0.0))
		bias = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(bias, name=scope.name)
		_activation_summary(conv1)
	
	# pool1
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], 
		strides=[1, 2, 2, 1],
		padding='SAME', name='pool1')

	# norm1
	# lcoal response normalization
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, 
		alpha=0.001 / 9.0, beta=0.75, name='norm1')

	# conv2
	with tf.variable_scope('conv2') as scope:
		kernel = _variable_with_weight_decay('weights',
			shape=[5, 5, 32, 64], stddev=1e-4, wd=0.0)
		conv = tf.nn.conv2d(norm1, kernel, 
			[1,1,1,1], padding="SAME")
		biases = _variable_on_cpu('biases', [64], 
			tf.constant_initializer(0.1))
		bias = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(bias, name=scope.name)
		_activation_summary(conv2)

	# norm2
	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, 
		alpha=0.001 / 9.0, beta=0.75, name='norm2')

	# pool2
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], 
		strides=[1, 2, 2, 1],
		padding='SAME', name='pool2')

	# local3
	with tf.variable_scope('local3') as scope:
		# reshape matrix to 1-D to facilitate compute
		dim = 1
		for d in pool2.get_shape()[1:].as_list():
			dim *= d
		reshape = tf.reshape(pool2, [BATCH_SIZE, dim])
		weights = _variable_with_weight_decay('weights',
			shape=[dim, 384], stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [384], 
			tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases,
			name=scope.name)
		_activation_summary(local3)

	# local4
	with tf.variable_scope('local4') as scope:
		weights = _variable_with_weight_decay('weights',
			shape=[384, 192], stddev=0.04, wd=0.004)
		biases = _variable_on_cpu('biases', [192],
			tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases,
			name=scope.name)
		_activation_summary(local4)

	# softmax
	with tf.variable_scope('softmax_linear') as scope:
		weights = _variable_with_weight_decay('weights',
			[192, NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
		biases = _variable_on_cpu('biases', [NUM_CLASSES],
			tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(local4, weights), biases,
			name=scope.name)
		_activation_summary(softmax_linear)

	return softmax_linear


def loss(logits, labels):
	# Add L2Loss to all the trainable variables
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, 
		name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	# The total loss = cross entropy + all decay terms
	return tf.add_n(tf.get_collection('losses'), 
		name='total_loss')


def _add_loss_summaries(total_loss):
	# Add summaries for losses in the model
	# Compute the average of all losses
	loss_averages = tf.train.ExponentialMovingAverage(
		0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	# Attach a scalar summary to losses
	for l in losses + [total_loss]:
		tf.summary.scalar(l.op.name + ' (raw)', l)
		tf.summary.scalar(l.op.name, loss_averages.average(l))

	return loss_averages_op

def train(total_loss, global_step):
	# Train the model
	num_batched_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
	decat_steps = int(num_batched_per_epoch * NUM_EPOCHS_PER_DECAY)

	# Decay the learning rate exponentially based on the num of step
	lr = tf.train.exponential_decay(
		INITIAL_LEARNING_RATE, global_step, decat_steps,
		LERANING_RATE_DECAY_FACTOR, staircase=True)
	tf.summary.scalar('learning_rate', lr)

	# Generate moving averages of all losses
	loss_averages_op = _add_loss_summaries(total_loss)

	# Compute gradients
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.GradientDescentOptimizer(lr)
		grads = opt.compute_gradients(total_loss)

	# Apply gradients
	apply_gradient_op = opt.apply_gradients(grads,
		global_step=global_step)

	# Add histograms for trainable variables:
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)

	# Add histograms for gradients
	for grad, var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name + '/gradients',
				grad)

	# Track the moving averages of all trainable variables
	variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	return train_op