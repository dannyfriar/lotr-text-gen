import sys
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib.keras.python.keras.utils import np_utils

enc = OneHotEncoder()
RESULTS_FILE = 'results/train_results.csv'
MODEL_FILE = 'models/lstm_text_model'
LOGS_PATH = 'logs'

#-------------- Misc functions
def progress_bar(value, endvalue, bar_length=20):
	percent = float(value) / endvalue
	arrow = '-' * int(round(percent * bar_length)-1) + '>'
	spaces = ' ' * (bar_length - len(arrow))
	sys.stdout.write("\rPercent complete: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
	sys.stdout.flush()

def one_hot_encode(idx, vocab_size=95):
	seq = [0] * vocab_size
	seq[idx] = 1
	return seq

#-------------- TF functions
def lstm_text(x, keep_prob, cell_size, n_hidden, learning_rate, seq_len, 
	num_layers, vocab_size):
	"""Build TF computation graph"""
	W_out = tf.get_variable("W_out", [cell_size, vocab_size],
							initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
	b_out = tf.get_variable("b_out", [vocab_size],initializer=tf.constant_initializer(0.01))

	# LSTM
	lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=cell_size) for _ in range(num_layers)]
	stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple=True)
	outputs, state = tf.nn.dynamic_rnn(cell=stacked_lstm, dtype=tf.float32, inputs=x)
	
	# Fully-connected
	outputs = tf.reshape(outputs, [-1, cell_size])
	outputs = tf.matmul(outputs, W_out) + b_out
	outputs = tf.reshape(outputs, [-1, seq_len, vocab_size])
	outputs = outputs[:, :-1, :]
	x_out = x[:, 1:, :]

	# Loss
	acc = tf.cast(tf.equal(tf.round(tf.sigmoid(outputs)), x_out), 'float')
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x_out, logits=outputs), 
						 reduction_indices=[0, 1])
	opt = tf.train.AdamOptimizer(learning_rate).minimize(tf.reduce_mean(loss))
	return loss, opt, acc


def main():
	#-------- Hyper-params
	seq_len = 50
	learning_rate = 0.002
	b_size = 50
	k_prob = 0.5
	cell_size = 256
	n_hidden = 128
	num_layers = 2
	n_epochs = 1
	print_freq = 200  # number of batches

	#-------- Load and prepare data
	print("#--------- Loading and preparing data...")
	with open('data/fellowship_text.txt', 'rb') as f:
		text_data = f.read()
	text_data = text_data.decode("utf-8")
	text_data.replace('\n', '')

	# Get chars and create dictionary lookups
	char_list = sorted(list(set(text_data)))
	vocab_size = len(char_list)
	n_chars = len(text_data)
	ix_to_char = {ix:char for ix, char in enumerate(char_list)}
	char_to_ix = {char:ix for ix, char in enumerate(char_list)}
	print("Vocab size = {} characters".format(len(char_list)))
	print("Data size = {} characters".format(n_chars))

	# Split into sequences
	data_x = []
	data_y = []
	for i in range(0, n_chars - seq_len, 1):
		seq_in = text_data[i:i + seq_len]
		seq_out = text_data[i + seq_len]
		data_x.append([char_to_ix[char] for char in seq_in])
		data_y.append(char_to_ix[seq_out])
		if i % 100000 == 0:
			print("Running for pattern {} of {}.".format(i, n_chars-seq_len))
	n_patterns = len(data_x)
	print("Total of {} patterns".format(n_patterns))

	# Fit sklearn encoder
	enc.fit(data_x[0:300000])


	#-------- Build TF graph and run TF session
	print("#--------- Building Tensorflow graph....")
	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [None, seq_len, vocab_size])
	keep_prob = tf.placeholder(tf.float32)
	loss, opt, acc = lstm_text(x, keep_prob, cell_size, n_hidden, learning_rate, seq_len, 
		num_layers, vocab_size)

	n_batches = int(len(data_x) / b_size)
	results_dict = {'epoch': [], 'batch': [], 'loss': []}
	saver = tf.train.Saver()
	writer = tf.summary.FileWriter(LOGS_PATH, graph=tf.get_default_graph())
	tf.summary.scalar("Cross Entropy Loss", loss)
	# tf.summary.scalar("Accuracy", acc)
	summary_op = tf.summary.merge_all()

	print("#--------- Training model...")
	print("Running with batch size {}, learning rate {} for {} epochs"\
		.format(b_size, learning_rate, n_epochs))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for epoch in range(n_epochs):
			t0 = time.time()
			print("#-------- Starting epoch {}, running for {} batches".format(epoch+1, n_batches))
			for batch in range(n_batches):
		#         progress_bar(n_batch+1, n_batches)
				sample = data_x[batch*b_size:(batch+1)*b_size]
				sample = np.array([[one_hot_encode(elem) for elem in s] for s in sample])
				l, o, a, s = sess.run([loss, opt, acc, summary_op], feed_dict={x: sample, keep_prob: k_prob})
				writer.add_summary(s, (epoch+1) * b_size + batch)
			
				if batch % print_freq == 0:
					print("Run for {} batches, loss = {}".format(batch, float(l)))
					results_dict['epoch'].append(epoch+1)
					results_dict['batch'].append(batch)
					results_dict['loss'].append(float(l))
					saver.save(sess, MODEL_FILE)
					pd.DataFrame.from_dict(results_dict).to_csv(RESULTS_FILE)
					
			epoch_time = time.time() - t0
			print("#--------- Finished run for epoch {} in time {}".format(epoch+1, epoch_time))
		
		sess.close()




if __name__ == "__main__":
	main()
