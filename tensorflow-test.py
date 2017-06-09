#!/usr/bin/env python
import tensorflow as tf


# The following will output 2 floating point tensors 
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print("\n\n *** Output nodes *** \n\n")
print(node1, node2)


# The following will create a Session object, and 
# run a computational graph on the 2 nodes created above
print("\n\n *** Create session an run graph *** \n\n")
sess = tf.Session()
print(sess.run([node1, node2]))


# Combine Tensor nodes with operations
print("\n\n *** Combining Tensor Nodes *** \n\n")
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))


# Using placeholders 
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

# Eveluate the above graph with multiple inputs 
# by using the feed_dict parameter
print("\n\n *** Evaluating the graph of placeholders *** \n\n")
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))


# Make comp graph more complex 
print("\n\n *** Make graph more complex by adding another operation *** \n\n")
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))


# modify the graph to get new outputs with the same input 
# Variables allow us to add trainable parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b


# Initialize all variables in a TensorFlow program
init = tf.initialize_all_variables()
sess.run(init)


print("\n\n *** Evaluate linear_model for several values of x  *** \n\n")
print(sess.run(linear_model, {x: [1,2,3,4]}))