import os
import glob
import numpy
numpy.random.seed(1234)
import tensorflow as tf
tf.set_random_seed(1234)

class BatchImageInput():
  def __init__(self, image_paths,labels, batch_size = 32):
    self.batch_size   = batch_size
    self.image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
    self.labels = tf.convert_to_tensor(labels, dtype=tf.int32)
  
  def read_one_file(self, filename_queue):
    # print filename_queue
    image_reader = tf.WholeFileReader()
    I  = tf.read_file(filename_queue[0])
    Id = tf.image.decode_image(I, channels=3)
    Id = tf.reshape(Id, [32,32,3])
    
    label = filename_queue[1]
    Id = tf.to_float(Id)
    return Id, label

  def get_minibatch_tensors(self, num_epochs=None):
    
    input_queue = tf.train.slice_input_producer([self.image_paths, self.labels], num_epochs=num_epochs, shuffle=True)

    Ic, label  = self.read_one_file(input_queue)
    
    Ic_batch, label_batch = tf.train.batch([Ic, label], batch_size=self.batch_size, capacity = 100)
    # (X,Y)
    return Ic_batch, label_batch


def get_image_label_from_list(line_list,path_prefix=None):
  content = map(lambda k: k.split(' '), line_list)
  if path_prefix is None:
    content = [(path, int(label)) for path, label in content ]
  else:
    content = [(os.path.join('dataset',path), int(label)) for path, label in content ]
  image_paths, labels = zip(*content)
  return image_paths, labels


if __name__ == '__main__':
  with open('dataset/cifar/train/train.txt', 'r') as f:
    file_paths = f.read().splitlines()[:1000]
    image_paths, labels = get_image_label_from_list(file_paths, path_prefix='dataset')

  EI = BatchImageInput(image_paths, labels, batch_size = 32)
  
  x_,y_ = EI.get_minibatch_tensors()

  with tf.Session() as sess:
    tf.global_variables_initializer()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    x,y = sess.run([x_,y_])
    print x[0],y[0]
    print x[0].shape
    coord.request_stop()
    coord.join(threads)
