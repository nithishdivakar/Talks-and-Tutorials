import numpy
numpy.random.seed(1234)
import tensorflow as tf
tf.set_random_seed(1234)

from tensorflow.examples.tutorials.mnist import input_data

from Model import Model
import Losses

import argparse

## -- begin --  command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lr"                ,type=float)
parser.add_argument("--log-interval"      ,type=int)
parser.add_argument("--snapshot-interval" ,type=int)
parser.add_argument("--batch-size"        ,type=int)
parser.add_argument("--hidden-neurons"    ,type=int)
parser.add_argument("--updates"           ,type=int)
args = parser.parse_args()
## -- end --  command line arguments


mnist = input_data.read_data_sets("dataset/", one_hot=True)

x_ = tf.placeholder(dtype = tf.float32, shape = [None, 28*28])
y_ = tf.placeholder(dtype = tf.float32, shape = [None, 10])

M = Model(hidden_neurons=args.hidden_neurons) 
optimizer = tf.train.GradientDescentOptimizer(args.lr)  
y_logits_ , _= M.inference(x_)
loss_ = Losses.cross_entropy(labels = y_, logits = y_logits_)
train_op_ = optimizer.minimize(loss_)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_logits_, 1))
accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  for step in range(0, args.updates):
    batch_x, batch_y = mnist.train.next_batch(args.batch_size)
    sess.run(
      [train_op_],
      feed_dict = {
        x_ : batch_x,
        y_ : batch_y,
      }
    )
    if step % args.log_interval == 0:
      loss,accuracy = sess.run(
               [loss_, accuracy_],
               feed_dict = {
                 x_ : mnist.test.images,
                 y_ : mnist.test.labels,
               }
             )
      print("Step {:6d} of {:6d}  Loss {:e} Accuracy {:4.2f}%".format(step+1, args.updates, loss, accuracy*100))
    if step % args.snapshot_interval == 0:
      saver.save(sess, "model.ckpt")
               
