import tensorflow as tf 
import numpy as np
import os
import glob

# Retrieve class name
def get_name(arr):
	mm = 'MM'
	mm_inh = 'MM Inh'

	if np.argmax(arr) == 0:
		return mm_inh
	elif np.argmax(arr) == 1:
		return mm

# Network parameters
hidden_size = 126
softmax_size = 86
learning_rate = 0.001
epochs = 20
max_len = 51
n_classes = 2
num_batches = 47
display_step = 1
test_batches = 4
save = False

# Placeholders
x = tf.placeholder(shape=[None, 2, 51], dtype=tf.float32)
y_true = tf.placeholder(shape=[None,2], dtype=tf.float32)

# Define model
def recurrent_net(x, used=False):
	with tf.variable_scope('Recurrent_Net'):
		# Unstack sequential data
		x = tf.unstack(x, max_len, axis=2)
		# Create cell
		if used is True:
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, reuse=True)
		else:
			lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
		# Run Recurrent network
		outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

		return outputs[-1], states[-1]

def mlp_classification(rnn_out):
	with tf.variable_scope('Logits'):
		# Define weights
		w_1 = tf.get_variable('w_1', [hidden_size, softmax_size], dtype=tf.float32)
		b_1 = tf.get_variable('b_1', [softmax_size], dtype=tf.float32)
		w_2 = tf.get_variable('w_2', [softmax_size, n_classes], dtype=tf.float32)
		b_2 = tf.get_variable('b_2', [n_classes], dtype=tf.float32)
		# Feed forward
		layer_1 = tf.nn.relu( tf.matmul(rnn_out, w_1) + b_1)
		layer_2 = tf.matmul(layer_1, w_2) + b_2

		return layer_2

# Evaluate model
out, state = recurrent_net(x)
logits = mlp_classification(out)

# Define cost and optimizer
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits) )
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(cost)
init = tf.global_variables_initializer()

# Run model
with tf.Session() as sess:
	print('\nRunning Session')
	init.run()
	os.chdir(input( 'Enter path to batch files: ' ))

	for epoch in range(epochs):
		used = []
		print("\n")
		for i in range(num_batches):
			batch_data = np.load('Batch_%i_Data.npy' % i)
			batch_classes = np.load('Batch_%i_Classes.npy' % i)
			# Run optimization op (backprop)  and cost op
			_, c = sess.run([training_op,cost], feed_dict={x: batch_data, y_true: batch_classes})
			# Display logs per epoch step
			if epoch % display_step == 0:
				print('     ','Epoch:', '%04d' % (epoch+1),
						'cost=' '{:.9f}'.format(c))

	print('\nOptimization Finished!\n')

	# Save model
	if save:
		saver = tf.train.Saver()
		save_path = saver.save(sess, "%smodel.ckpt" % input('Enter path to save folder: ')
		print("Model saved in file: %s\n" % save_path)

	# Test model
	os.chdir('Enter path to test-batch files: ')
	
	print('Testing model\n')
	
	false, true = 0, 0
	
	for i in range(1,test_batches+1,1):
		print('     ', 'Test-Batch', i)
		test_batch_data = np.load('Test_Batch_%i_Data.npy' % i)
		test_batch_classes = np.load('Test_Batch_%i_Classes.npy' % i)
		for inst in range(test_batch_data.shape[0]):
			instance = np.expand_dims(test_batch_data[inst, :, :], axis=0)
			true_class = get_name(test_batch_classes[inst, :])
			pred_class = sess.run(tf.nn.softmax(logits), feed_dict={x:instance})
			pred_class = get_name(pred_class)

			if pred_class == true_class:
				true += 1
			elif pred_class != true_class:
				false += 1

	print('\nAccuracy:', true/(true+false), 'Failure:', false/(true+false))

	with open('Accuracy_Log.txt', "w") as file:
		file.write('Accuracy: %f' % true/(true+false))
