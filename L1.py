from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import utils
import datetime

starttime = datetime.datetime.now()
#######################
# Model Configuration #
#######################
tf.app.flags.DEFINE_float('base_lr', 0.03, 'initialized learning rate')
tf.app.flags.DEFINE_float('stepsize', 5000, '')
tf.app.flags.DEFINE_float('decay_rate', 0.9, '')
tf.app.flags.DEFINE_float('memory_usage', 0.94, '')
tf.app.flags.DEFINE_integer('train_display', 100, '')
tf.app.flags.DEFINE_integer('test_iter', 1000, '')
tf.app.flags.DEFINE_integer('max_iter', 30000, '')

#############################
# Regularizer Configuration #
#############################
tf.app.flags.DEFINE_float('lamb', 0.00005, 'regularizer parameter')
tf.app.flags.DEFINE_boolean('cges', True, 'Combined group and exclusive sparsity')
#tf.app.flags.DEFINE_float('a', 0.01, 'TL1 parameter')

######################
# CGES Configuration #
#t#####################
#        tf.app.flags.DEFINE_float('base_lr', 5*pow(10,-p), 'initialized learning rate')
#        tf.app.flags.DEFINE_float('lamb', 5*pow(10,-j), 'regularizer parameter')
FLAGS = tf.app.flags.FLAGS
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.memory_usage)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784]) # single flattened 28 * 28 pixel MNIST image
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # 10 classes output
keep_prob = tf.placeholder(tf.float32)

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10])) 

batch = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
       FLAGS.base_lr,      # Base learning rate.
       batch,              # Current index. 
       FLAGS.stepsize,     # Decay iteration step. 
       FLAGS.decay_rate,   # Decay rate. 
       staircase=True)  

from mnist_model import mnist_conv
y_conv = mnist_conv(x, 10, keep_prob)

S_vars = [svar for svar in tf.trainable_variables() if 'weight' in svar.name]
ff_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_)) 
if not FLAGS.cges:
    ff_loss_reg = ff_loss + learning_rate * 0.01 * \
                tf.reduce_sum([tf.nn.l2_loss(var) for var in S_vars])
else:
    ff_loss_reg = ff_loss

opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(ff_loss_reg, global_step=batch)
#opt = tf.train.AdamOptimizer(0.0001).minimize(ff_loss_reg, global_step=batch)

sess.run(tf.global_variables_initializer())

op_list = []
if FLAGS.cges:
    # Normalization parameter
    #glayerwise = [0.1, 0.1, 1./5, 1./2]
    #elayerwise = [0.1, 0.1, 5., 2.]  
    #glayerwise = [1./2, 1./2, 1./5, 1./2]
    elayerwise = [1, 1, 5., 10.]
    #glayerwise = [1.,1.0, 1./15, 1./144]
    #elayerwise = [1.,0.5, 15., 144.]
    for vind, var in enumerate(S_vars):
        e_param = learning_rate * FLAGS.lamb
#        t_stack = tf.stack([t_ for _ in range(gl_op.get_shape()[-1])], -1)
        gl_plus = tf.cast((tf.abs(var)-e_param * elayerwise[vind])> 0, tf.float32) * (tf.abs(var)-e_param* elayerwise[vind])
        GLS = tf.sign(var) * gl_plus
        GLS_op = var.assign(GLS)
        op_list.append(GLS_op)     
        

    with tf.control_dependencies(op_list):
        cges_op_list = tf.no_op()

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


save_sparsity = []
f=open(r'F:\\result\mnist2\l1\using_param1.txt','w')
f2=open(r'F:\\result\mnist2\l1\test_accuracy1.txt','w')
f3=open(r'F:\\result\mnist2\l1\FLOP1.txt','w')
for i in range(30000):
#for i in range(FLAGS.max_iter):
    batch = mnist.train.next_batch(100)

    # Display
    if (i+1) % FLAGS.train_display == 0:
        train_accuracy, tr_loss = sess.run([accuracy, ff_loss], \
                feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, lr %.4f, training accuracy %g" \
                %(i+1, sess.run(learning_rate), train_accuracy))

        ratio_w, sp = utils._comp(S_vars)
        _sp = sess.run(sp)

        print("loss: %.4f sp: %0.4f %0.4f %0.4f %0.4f :: using param : %.4f" \
                            %(tr_loss, _sp[0], _sp[1], _sp[2], _sp[3], sess.run(ratio_w)))
        print(sess.run(ratio_w),file=f)
        
    # Training
    opt.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5}) 
    if FLAGS.cges:
        _ = sess.run(cges_op_list)
        
    # Testing
    if (i+1) % FLAGS.test_iter == 0:
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, \
                y_: mnist.test.labels, keep_prob: 1.0})
        print("test accuracy %0.4f" %(test_acc))
        print(test_acc,file=f2)

        # Computing FLOP
        flop = utils._cost(_sp)
        print("FLOP : %.4f" %(flop))
        print(flop,file=f3)

        flop = utils._cost(_sp)
        if FLAGS.cges:
            print('CGES, lambda : %f' \
                        %(FLAGS.lamb))

f.close()
f2.close()
f3.close()

endtime = datetime.datetime.now()
print (endtime-starttime)

from visualization import vis_square
filters=[]
filters=S_vars[0]
fil=sess.run(filters)
filter2=fil.reshape([5,5,32])
vis_square(filter2.transpose(2,0,1))
