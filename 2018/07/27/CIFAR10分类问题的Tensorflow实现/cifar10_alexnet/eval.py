import tensorflow as tf
from datetime import datetime
import time
import math
import numpy as np 
import model

EVAL_DIR = './cifar10_eval'
EVAL_DATA = 'test'
EVAL_INTERVAL_SECS = 60 * 5
RUN_ONCE = False

CHECKPOINT_DIR = model.TRAIN_DIR
NUM_EXAMPLES = model.NUM_EXAMPLES_PER_EVAL
BATCHI_SIZE=model.BATCHI_SIZE

def eval_once(saver, summary_writer, top_k_op, summary_op):
	# Run Eval once
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			print('No checkpoint file found')
			return

	# Start queue runners
	coord = tf.train.Coordinator()
	try:
		threads = []
		for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
			threads.extend(qr.create_threads(sess, coord=coord,
				daemon=True, start=True))

		num_iter = int(math.ceil(NUM_EXAMPLES / BATCHI_SIZE))
		true_count = 0
		total_sample_count = num_iter * BATCHI_SIZE
		step = 0
		while step < num_iter and not coord.should_stop():
			predictions = sess.run([top_k_op])
			true_count += np.sum([predictions])
			step += 1

		# Compute precision @ 1
		precision = true_count / total_sample_count
		print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

		summary = tf.Summary()
		summary.ParseFromString(sess.run(summary_op))
		summary.value.add(tag='Precision @1', simple_value=precision)
		summary_writer.add_summary(summary, global_step)
	except Exception as e:
		coord.request_stop(e)

	coord.request_stop()
	coord.join(threads, stop_grace_period_secs=10)


def evaluate()
	# Eval model for a num of steps
	with tf.Graph().as_default():
		eval_data = EVAL_DATA == 'test'	
		images, labels = model.inputs(eval_data=eval_data)

		# Build a Graph to compute the logits predictions
		# from the inference model
		logits = model.inference(images)

		# Calc predicions
		top_k_op = tf.nn.in_top_k(logits, labels, 1)

		# Restore the moving average version of learned variables
		variables_averages = tf.train.ExponentialMovingAverage(
			model.MOVING_AVERAGE_DECAY)
		variables_to_restore = variables_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		# Build summart op
		summary_op = tf.summary.merge_all()
		graph_def = tf.get_default_graph().as_graph_def()
		summary_writer = tf.summary.FileWriter(EVAL_DIR,
			graph_def=graph_def)

		while True:
			eval_once(saver, summary_writer, top_k_op, summary_op)
			if RUN_ONCE:
				break
			time.sleep(EVAL_INTERVAL_SECS)

evaluate()