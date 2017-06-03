import numpy
numpy.random.seed(1234)
import tensorflow as tf
tf.set_random_seed(1234)


from Model import Model
from Input import BatchImageInput, get_image_label_from_list
import Losses

import argparse
import sys

## -- begin --  command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lr"                ,type=float)
parser.add_argument("--log-interval"      ,type=int)
parser.add_argument("--snapshot-interval" ,type=int)
parser.add_argument("--batch-size"        ,type=int)
parser.add_argument("--updates"           ,type=int)
args = parser.parse_args()
## -- end --  command line arguments

with open('dataset/cifar/train/train.txt', 'r') as f:
  file_paths = f.read().splitlines()
  image_paths, labels = get_image_label_from_list(file_paths, path_prefix='dataset')

train_fraction = 0.8
index = int(len(image_paths)*train_fraction)

TrainData = BatchImageInput(image_paths[:index], labels[:index], batch_size = args.batch_size)
ValdnData = BatchImageInput(image_paths[index:], labels[index:], batch_size = args.batch_size)



x_train_,y_train_ = TrainData.get_minibatch_tensors()
x_valdn_,y_valdn_ = ValdnData.get_minibatch_tensors()

phase_train_ = tf.placeholder(dtype = tf.bool,shape=[])

x_ = tf.where(phase_train_, x_train_, x_valdn_)
y_ = tf.where(phase_train_, y_train_, y_valdn_)



M = Model()
optimizer = tf.train.AdamOptimizer(args.lr)
y_logits_ , _= M.inference(x_)
loss_ = Losses.cross_entropy(labels = y_, logits = y_logits_)
train_op_ = optimizer.minimize(loss_)

correct_prediction = tf.equal(y_, tf.cast(tf.argmax(y_logits_, 1),dtype=tf.int32))
accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

###
tf.summary.scalar('loss',loss_)

tf.summary.image('input',x_)

###
summary_op = tf.summary.merge_all()




with tf.Session() as sess:
  ###
  writer = tf.summary.FileWriter("./", graph=sess.graph)

  init = tf.global_variables_initializer()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
 
  sess.run(init)
  for step in range(0, args.updates):
    ###
    summary = sess.run( [summary_op], feed_dict={ phase_train_:True})

    sess.run(
      [train_op_],
      feed_dict = {
        #x_: x_train_batch,
        #y_: y_train_batch,
        phase_train_:True,
      }
    )

    writer.add_summary(summary[0], step)

    if step % args.log_interval == 0:
      loss,accuracy = sess.run(
               [loss_, accuracy_],
               feed_dict = {
                 #x_: x_valdn_batch,
                 #y_: y_valdn_batch,
                 phase_train_: False,
               }
             )
      print("Step {:6d} of {:6d}  Loss {:e} Accuracy {:4.2f}%".format(step+1, args.updates, loss, accuracy*100))
      sys.stdout.flush()
    if step % args.snapshot_interval == 0:
      saver.save(sess, "model.ckpt")
  coord.request_stop()
  coord.join(threads)
