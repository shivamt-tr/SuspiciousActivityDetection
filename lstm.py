# -*- coding: utf-8 -*-
"""
    Final_Year_Project.ipynb
"""

import numpy as np
from sklearn.model_selection import train_test_split


'''
    IV. Preparing the Inception v3 output for input to the LSTM
'''

'''
Shuffle and split the data into the train and test set.
'''

# Load saved features extracted from Inception v3
inception_features = np.load('inception_features.npy')
inception_labels = np.load('inception_labels.npy')

# Split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(inception_features,
                                                    inception_labels,
                                                    test_size=0.20,
                                                    random_state=42)

# Normalize the training data
X_train /= np.max(X_train)

# %%


'''
    V. Training the LSTM network.
'''

"""
    Recurrent Neural Network.
    A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
    Links:
        [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
        Author: Aymeric Damien
        Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

import tensorflow as tf
from tensorflow.contrib import rnn

'''
    To classify the inception's features using a recurrent neural network,
    we will consider the 15x51200 vector of features. The input will
    be considered as 15 sequences of 512000 inputs each.
'''

# Training Parameters
learning_rate = 0.001
training_steps = 200
batch_size = 10
display_step = 10

# Network Parameters
n_input = 51200 # Data input (Features shape: 15x51200)
timesteps = 15 # Timesteps
n_hidden1 = 256 # hidden layer num of features
n_hidden2 = 100 # hidden layer num of features
n_classes = 2 # Total classes (Violent and Non-Violent)

# Reset the default graph in TensorFlow
tf.reset_default_graph()

# Declare tensors for data input and network output
X = tf.placeholder("float", [None, timesteps, n_input])
Y = tf.placeholder("float", [None, n_classes])
# keep_prob = tf.placeholder(tf.float32)

# Define weights and biases for network
weights = {
        'hidden': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
        'output': tf.Variable(tf.random_normal([n_hidden2, n_classes]))
        }

biases = {
        'hidden_b': tf.Variable(tf.random_normal([n_hidden2])),
        'output_b': tf.Variable(tf.random_normal([n_classes]))
        }


'''
    RNN(x, weights, biases):
        Input:
            x - Input Data
            weights - Randomly initialized weights for network layers
            biases - Randomly initialized biases for network layers
        Output:
            The function RNN runs the input through the network and
            outputs 2 values from the output layer.
'''
def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a LSTM cell with TensorFlow
    lstm_cell = rnn.BasicLSTMCell(n_hidden1, forget_bias=1.0)

    # Get LSTM cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    
    # Add Drop-Out with keep_prob=0.5
    drop_out = tf.nn.dropout(outputs[-1], keep_prob=0.5)

    # Feedforward through the hidden layer
    hidden_layer_output = tf.nn.sigmoid(tf.add(
                                        tf.matmul(drop_out, weights['hidden']),
                                        biases['hidden_b']))
    
    out = tf.nn.sigmoid(tf.add
                        (tf.matmul(hidden_layer_output, weights['output']),
                         biases['output_b']))

    return out

# %%

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss (softmax cross entropy) and optimizer (gradient descent)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):

        # Select 'batch_size' training samples from the data
        max_batch_num = X_train.shape[0] / batch_size  # Number of batches for given batch_size
        idx = int((step-1) % max_batch_num)  # Starting index of the current batch slice
        batch_x = X_train[idx*10:(idx+1)*10]  # Slice training data features
        batch_y = y_train[idx*10:(idx+1)*10]  # Slice training data labels

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        if step % display_step == 0 or step == 1:

            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for test data
    test_data, test_label = X_test, y_test
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
