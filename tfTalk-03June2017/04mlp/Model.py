import tensorflow as tf

class Model(object):
  def __init__(self,hidden_neurons=10):
    self.number_hidden_neurons = hidden_neurons
    self.Theta = {}
    self.init_model()


  def init_model(self):
    def add_param(shape,name):
      self.Theta[name] = tf.Variable(tf.truncated_normal(shape = shape, mean=0.0,stddev=0.01), name = name)

    add_param([28*28, self.number_hidden_neurons], 'W1')
    add_param([self.number_hidden_neurons], 'b1')
    
    add_param([self.number_hidden_neurons,10], 'W2')
    add_param([10], 'b2')

  def inference(self, input_):
    wx1 = tf.matmul(input_, self.Theta['W1'])
    z1 = tf.nn.bias_add(wx1, self.Theta['b1'])
    a1 = tf.nn.relu(z1)

    wx2 = tf.matmul(a1, self.Theta['W2'])
    z2 = tf.nn.bias_add(wx2, self.Theta['b2'])
    a2 = tf.nn.softmax(z2)

    return z2, a2
