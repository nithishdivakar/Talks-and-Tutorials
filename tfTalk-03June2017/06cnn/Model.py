import numpy
numpy.random.seed(1234)
import tensorflow as tf
tf.set_random_seed(1234)


class Model(object):
  def __init__(self):
    self.Theta = {}
    self.init_model()

  def get_training(self):
    return self.training

  def set_training(self, training):
    self.training = tf.assign(self.training, tf.constant(training, shape=[]))
  
  def init_model(self):
    def add_param(shape,name):
      self.Theta[name] = tf.Variable(tf.truncated_normal(shape = shape, mean=0.0,stddev=0.01), name = name)
    add_param([3,3,  3, 64],'w1')
    add_param([         64],'b1')
    add_param([3,3, 64,128],'w2')
    add_param([        128],'b2')
    add_param([3,3,128,256],'w3')
    add_param([        256],'b3')
    add_param([3,3,256,512],'w4')
    add_param([        512],'b4')

    add_param([4*4*512,1024],'w5')
    add_param([       1024],'b5')

    add_param([   1024, 10],'w6')
    add_param([         10],'b6')
 
  def inference(self,image_):

    def conv(ind,image_,stride=1):
      wx = tf.nn.conv2d(image_, self.Theta['w{}'.format(ind)], strides=[1, stride, stride, 1], padding='SAME',name="w{}".format(ind))
      z  = tf.nn.bias_add(wx, self.Theta['b{}'.format(ind)], name="b{}".format(ind))
      return z

    def dense(ind,input_):
      wx = tf.matmul(input_, self.Theta['w{}'.format(ind)])
      z  = tf.nn.bias_add(wx, self.Theta['b{}'.format(ind)])
      return z
 

    z1 = conv(1,image_)
    a1 = tf.nn.relu(z1)

    z2 = conv(2,a1,stride=2)
    a2 = tf.nn.relu(z2)

    z3 = conv(3,a2,stride=2)
    a3 = tf.nn.relu(z3)

    z4 = conv(4,a3,stride=2)
    a4 = tf.nn.relu(z4)
    sh = a4.get_shape().as_list()

    ff = tf.reshape(a4, [-1, numpy.prod(sh[1:])])
    
    z5 = dense(5,ff)
    a5 = tf.nn.relu(z5)
     
    z5 = dense(6,a5)
    a5 = tf.nn.softmax(z5)
    
    return a5,z5
    
