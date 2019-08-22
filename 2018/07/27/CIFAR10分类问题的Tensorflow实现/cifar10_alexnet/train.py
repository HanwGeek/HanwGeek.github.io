import tensorflow as tf
#from tensorflow.python import debug as tfdbg
from datetime import datetime
import model
import time
import os

TRAIN_DIR = model.TRAIN_DIR
MAX_STEP = 100000
LOG_DEVICE_PLACEMENT = False
batch_size = model.BATCH_SIZE

def train():
	# Train model for a num of steps
	with tf.Graph().as_default():
		global_step = tf.Variable(0, trainable=False)

		# Get images
		images, labels = model.distorted_inputs()

		# Build a Graph to compute the logits prediction
		# from inference model
		logits = model.inference(images)

		# Calc loss
		loss = model.loss(logits, labels)

		# Build a Graph to train the model and update
		# the model parameters
		train_op = model.train(loss, global_step)

		# Create a saver
		saver = tf.train.Saver(tf.all_variables())

		# Build the summary op
		summary_op = tf.summary.merge_all()

		# Build an initial op
		init = tf.initialize_all_variables()

		# Start running op
		sess = tf.Session(config=tf.ConfigProto(
			log_device_placement=LOG_DEVICE_PLACEMENT))
		sess.run(init)

		# Start the queue runners
		tf.train.start_queue_runners(sess=sess)
		summary_writer = tf.summary.FileWriter(TRAIN_DIR)

		for step in range(MAX_STEP):
			start_time = time.time()
			_, loss_val = sess.run([train_op, loss])
			duration = time.time() - start_time

			if step % 10 == 0:
				num_examples_per_step = batch_size
				examples_per_sec = num_examples_per_step / duration
				#sec_per_examples = duration / num_examples_per_step
				sec_per_batch = float(duration)

			format_str = ('%s: step %d, loss = %.2f (%.1f examples_per_sec)')
			print(format_str % (datetime.now(), step, loss_val, examples_per_sec))

			if step % 100 == 0:
				summary_str = sess.run(summary_op)
				summary_writer.add_summary(summary_str, step)

			# Save the model checkpoint
			if step % 1000 == 0 or (step + 1) == MAX_STEP:
				checkpoint_path = os.path.join(TRAIN_DIR,
				 'model.ckpt')
				saver.save(sess, checkpoint_path, 
					global_step=global_step)

train()