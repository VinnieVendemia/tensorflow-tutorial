# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

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


# OUR MODEL
# Multiple x and W, add b, and finally 
# apply the softmax expression
y = tf.nn.softmax(tf.matmul(x, W) + b)


# In order to train the model, you want to define what it means for the model to 
# be bad.  We call this the cost/or loss. 
# 
# A nice function to determine loss is called 'cross-entropy'. 


# To implement cross-entropy, we need a placeholder to 
# input correct answers
y_ = tf.placeholder(tf.float32, [None, 10])


# We can then implement the cross-entropy function
# First, tf.log computes the logarithm of each element of y. 
# Next, we multiply each element of y_ with the corresponding element of tf.log(y). 
# Then tf.reduce_sum adds the elements in the second dimension of y, 
# due to the reduction_indices=[1] parameter. 
# Finally, tf.reduce_mean computes the mean over all the examples in the batch.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# Now that we know what we want our model to do, its very easy to have TensorFlow 
# train it to do so. Because TensorFlow knows the entire graph of your computations, 
# it can automatically use the backpropagation algorithm to efficiently determine how 
# your variables affect the loss you ask it to minimize. Then it can apply your choice 
# of optimization algorithm to modify the variables and reduce the loss.
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

# launch the model in an interactive session
# sess = tf.InteractiveSession()
