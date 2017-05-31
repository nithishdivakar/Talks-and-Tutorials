import numpy
numpy.random.seed(1234)
import tensorflow as tf
tf.set_random_seed(1234)

from tensorflow.examples.tutorials.mnist import input_data

from Model import Model
from Input import BatchImageInput, get_image_label_from_list
import Losses
import sys

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

with open('dataset/cifar/train/train.txt', 'r') as f:
  file_paths = f.read().splitlines()
  image_paths, labels = get_image_label_from_list(file_paths, path_prefix='dataset')

EI = BatchImageInput(image_paths, labels, batch_size = args.batch_size)
  
x_,y_ = EI.get_minibatch_tensors()
# x_ = tf.placeholder(dtype = tf.float32, shape = [None, 32*32])
# y_ = tf.placeholder(dtype = tf.float32, shape = [None, 10])

M = Model()
optimizer = tf.train.GradientDescentOptimizer(args.lr)  
y_logits_ , _= M.inference(x_)
loss_ = Losses.cross_entropy(labels = y_, logits = y_logits_)
train_op_ = optimizer.minimize(loss_)

correct_prediction = tf.equal(y_, tf.cast(tf.argmax(y_logits_, 1),dtype=tf.int32))
accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
 
  sess.run(init)
  for step in range(0, args.updates):
    sess.run(
      [train_op_],
      #feed_dict = {
      #  x_ : batch_x,
      #  y_ : batch_y,
      #}
    )
    if step % args.log_interval == 0:
      loss,accuracy = sess.run(
               [loss_, accuracy_],
               #feed_dict = {
               #  x_ : mnist.test.images,
               #  y_ : mnist.test.labels,
               #}
             )
      print("Step {:6d} of {:6d}  Loss {:e} Accuracy {:4.2f}%".format(step+1, args.updates, loss, accuracy*100))
      sys.stdout.flush()
    if step % args.snapshot_interval == 0:
      saver.save(sess, "model.ckpt")
  coord.request_stop()
  coord.join(threads)             
