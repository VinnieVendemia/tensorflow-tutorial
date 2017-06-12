from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# x is A placeholder value, we'll input when 
# we ask Tensorflow to run a computation.  We 
# want to be able to input any number of images flattened 
# into 784-dimensional vector.  This gets represented as a 
# 2-d floating point number with a shape [None, 784].  None 
# means the dimension can have any length
x = tf.placeholder(tf.float32, [None, 784])


# These represent the weights and bias of our model
# These variables are modifiable tensors that live in 
# the graph of interacting operations. They can be used 
# and even modified by the computation.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))