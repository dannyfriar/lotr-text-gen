import sys
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib.keras.python.keras.utils import np_utils

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

#------ Network to generate text
def test_lstm(x_first, cell_size, vocab_size, des_len, num_layers):
	"""Build computation graph for NN to generate text"""
	W_out = tf.get_variable("W_out", [cell_size, vocab_size],
							initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
	b_out = tf.get_variable("b_out", [vocab_size],initializer=tf.constant_initializer(0.01))
	predicted_probs = []
	
	# First LSTM
	lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=cell_size) for _ in range(num_layers)]
	stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple=True)
	outputs, state = tf.nn.dynamic_rnn(cell=stacked_lstm, dtype=tf.float32, inputs=x_first)

	outputs = tf.reshape(outputs, [-1, cell_size])
	outputs = tf.matmul(outputs, W_out) + b_out
	outputs = tf.reshape(outputs, [-1, 1, vocab_size])
	outputs = tf.nn.softmax(tf.reshape(outputs[:, -1, :], [-1, 1, vocab_size]), dim=-1)
	predicted_probs.append(outputs)
	
	# Feed this output until we've generated des_len characters
	tf.get_variable_scope().reuse_variables()
	for i in range(des_len-1):
		outputs, state = tf.nn.dynamic_rnn(cell=stacked_lstm, dtype=tf.float32, 
										   inputs=tf.round(outputs), initial_state=state)
		outputs = tf.reshape(outputs, [-1, cell_size])
		outputs = tf.matmul(outputs, W_out) + b_out
		outputs = tf.reshape(outputs, [-1, 1, vocab_size])
		outputs = tf.nn.softmax(tf.reshape(outputs[:, -1, :], [-1, 1, vocab_size]), dim=-1)
		predicted_probs.append(outputs)
		
	return predicted_probs

def main():
	#-------- Hyper-params
	vocab_size = 95
	des_len = 100
	cell_size = 64
	num_layers = 3

	#--------- Load and prepare data
	print("Loading and preparing vocabulary...")
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

	#-------- Create TF graph
	tf.reset_default_graph()
	x_first = tf.placeholder(tf.float32, [None, 1, vocab_size])
	predicted_probs = test_lstm(x_first, cell_size, vocab_size, des_len, num_layers)

	#-------- Generate text for a test character
	test_char = 't'
	print("Generating text for test character {}...".format(test_char))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		print("Reloading model...")
		saver = tf.train.import_meta_graph('models/lstm_text_model.meta')
		saver.restore(sess, tf.train.latest_checkpoint('models/'))
		all_vars = tf.get_collection('vars')
		
		test_array = np.array(one_hot_encode(char_to_ix[test_char])).reshape(1, 1, vocab_size)
		pixel_probs = sess.run(predicted_probs, feed_dict={x_first: test_array})

		sess.close()

	output_string = ''.join([ix_to_char[np.argmax(a)] for a in pixel_probs])
	print("Test output sentence = \n{}".format(output_string))


if __name__ == "__main__":
	main()