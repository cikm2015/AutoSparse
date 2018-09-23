# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer 

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')#定义了一个函数，来指定做卷积的步长为1，边缘直接复制过来

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def mnist_conv(x, num_classes, keep_prob):
  #filter1 = tf.get_variable("weight1", shape=[3, 3, 1, 32],initializer=tf.contrib.layers.xavier_initializer())
  filter1 = tf.get_variable("weight1", shape=[5, 5, 1, 32],initializer=tf.contrib.layers.xavier_initializer())
  #filter1 = tf.get_variable("weight1", shape=[3, 3, 1, 32],initializer=tf.contrib.layers.xavier_initializer())
  bias1 = tf.get_variable("bais1", [32], initializer=tf.contrib.layers.xavier_initializer())
  #filter1 = weight_variable([3, 3, 1, 32], 1)
  #bias1 = bias_variable([32], 1)
  
  x_ = tf.reshape(x, [-1, 28, 28, 1])
  
  relu1 = tf.nn.relu(conv2d(x_, filter1) + bias1)
  pool1 = max_pool_2x2(relu1)
  norm1 = tf.contrib.layers.batch_norm(pool1)
  #drop1 = tf.nn.dropout(pool1, 0.1)
    
  #filter2 = tf.get_variable("weight2", shape=[3, 3, 32, 64],initializer=tf.contrib.layers.xavier_initializer())
  filter2 = tf.get_variable("weight2", shape=[5, 5, 32, 64],initializer=tf.contrib.layers.xavier_initializer())
  #filter2 = tf.get_variable("weight2", shape=[3, 3, 32, 64],initializer=tf.contrib.layers.xavier_initializer())
  bias2 = tf.get_variable("bais2", [64], initializer=tf.contrib.layers.xavier_initializer())
  
  #filter2 = weight_variable([3, 3, 32, 64], 2)
  #bias2 = bias_variable([64], 2)

  relu2 = tf.nn.relu(conv2d(norm1, filter2) + bias2)
  pool2 = max_pool_2x2(relu2)
  norm2 = tf.contrib.layers.batch_norm(pool2)
  drop2 = tf.nn.dropout(norm2, 0.25)

  fc3 = tf.get_variable("weight3", shape=[7 * 7 * 64,1024],initializer=tf.contrib.layers.xavier_initializer())
  bias3 = tf.get_variable("bais3", [1024], initializer=tf.contrib.layers.xavier_initializer())
  
  #fc3 = weight_variable([7 * 7 * 64, 512], 3)
  #bias3 = bias_variable([512], 3)

  flat_pool2 = tf.reshape(drop2, [-1, 7 * 7 * 64])
  relu3 = tf.nn.relu(tf.matmul(flat_pool2, fc3) + bias3)

  drop3 = tf.nn.dropout(relu3, keep_prob)
  
  fc4 = tf.get_variable("weight4", shape=[1024, 10],initializer=tf.contrib.layers.xavier_initializer())
  bias4 = tf.get_variable("bais4", [10], initializer=tf.contrib.layers.xavier_initializer())

  output = tf.matmul(drop3, fc4) + bias4

  return tf.nn.softmax(output)

  
  

