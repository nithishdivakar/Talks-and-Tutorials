import argparse
parser = argparse.ArgumentParser(description='Expert denoisers training')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', required=True, help='input batch size for training (default: 32)')
parser.add_argument('--stride', type=int, default=16, metavar='N', required=True, help='input batch size for training (default: 32)')
parser.add_argument('--noise-level', type=str, default=5.0, metavar='N', help='No of iterations')
parser.add_argument('--model-snapshot-file', type=str, required=True, help='input batch size for training (default: 32)')
parser.add_argument('--test-data-list-file', type=str, metavar='N', required=True, help='test base folder is required')
parser.add_argument('--save-images', type=str, metavar='N', required=True, help='save images')
parser.add_argument('--quality', type=int, metavr='N', required=True, help='compression quality')
args = parser.parse_args()



import tensorflow as tf
import os
import numpy
import glob
from itertools import product
from PIL import Image
import skimage
from skimage.util import view_as_windows
import json
import  skimage.measure as measure
import sys
import importlib
import cPickle as pickle
from tqdm import tqdm

from Model import Model

try:
  tf.Session()
except: pass



def make_dirs(dirs):
  for _dir in dirs:
    if not os.path.isdir(_dir):
      os.mkdir(_dir)

def reconstruct_image_from_patches(Pp_p,s, In_p):
  Op = numpy.zeros(s)
  i=0
  for x in range(Op.shape[0]):
      for y in range(Op.shape[1]):
          Op[x,y] = Pp_p[i].reshape((s[2],s[3]))
          i = i+1

  Ip_p = numpy.zeros_like(In_p)
  Ip_w = numpy.zeros_like(In_p)


  for x in range(Op.shape[0]):
      for y in range(Op.shape[1]):
          Ip_p[x*stride[0]:(x)*stride[0]+64,y*stride[1]:(y)*stride[1]+64] += Op[x,y]
          Ip_w[x*stride[0]:(x)*stride[0]+64,y*stride[1]:(y)*stride[1]+64] += 1.0
          #psnr[x*stride:(x+1)*stride,y*stride:(y+1)*stride] = 2*sqrt(mean_squared_error(Op[x,y], Pc[x,y]))
  Ip_w[Ip_w==0.0] = 1.0
  Ip_p = Ip_p/Ip_w
  
  return Ip_p[32:-32,32:-32] # removing padding


def denoise_image(In, stride):
  In_p = skimage.util.pad(In,32, 'reflect')
  Pn_p = view_as_windows(In_p,window_shape=(64,64),step=stride)
  ss   = Pn_p.shape
  Pn_p = Pn_p.reshape(ss[0]*ss[1],ss[2],ss[3],1)/255.0
  Pp_p = numpy.zeros_like(Pn_p)
  
  for ind in range(0,ss[0]*ss[1],batch_size):
    #print ind,ind+batch_size
    mini_batch = Pn_p[ind:ind+batch_size]
    #print mini_batch.shape
    predk = E.sess.run([y_],feed_dict={x_:mini_batch, phase_train:False})
    #print numpy.asarray(predk).shape
    Pp_p[ind:ind+batch_size,:,:,:] = predk[0]
  Pp_p = Pp_p*255.0
  Ip   = reconstruct_image_from_patches(Pp_p,ss, In_p)
  return post_process(Ip)

def post_process(I):
  # clip values outside range
  I[I>255.0]=255.0
  I[I<  0.0]=  0.0
  return I

def lp_loss(y_true, y_pred):
  loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=y_pred, labels=y_true))
  return loss

def read_image(file_name):
  return numpy.asarray(Image.open(file_name).convert("L"),dtype='float32')

def jpeg_compress(I, quality):
  return tf.image.decode_jpeg(tf.image.encode_jpeg(I , quality = quality),channels = 3) 

def img_generator(quality, test_data_file):
  f = open(test_data_file,'r')
  for l in f.readlines():
    l = l.strip()
    I  = read_image(l)
    In = jpeg_compress(I, quality)
    yield I, In, l, l

generator    = img_generator(args.quality, args.test_data_list_file)
batch_size   = args.batch_size 
stride       = (args.stride,args.stride)

LG ='E2'

log_dir = os.path.join(LG, 'LOG')
# clean_images_save_folder = os.path.join(args.exp_code, 'save', 'clean_images')
# noisy_images_save_folder = os.path.join(args.exp_code, 'save', 'noise_std_{}'.format(args.noise_level))
reconstructed_images_save_folder = os.path.join(LG, 'save', 'recon_std_{}'.format(args.noise_level))

# make_dirs([args.exp_code, log_file, os.path.join(args.exp_code, 'save'),
#   clean_images_save_folder, noisy_images_save_folder])

make_dirs([LG, log_dir, os.path.join(LG, 'save'), reconstructed_images_save_folder])

#config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25))
config = tf.ConfigProto(
  device_count = {'GPU': 0}
)

with tf.Session(config=config) as sess:
  optimizer = tf.train.AdamOptimizer(1e-3)
  E = Model(sess,optimizer=optimizer, batch_size = batch_size, tot_res_blocks=16)
  x_ = tf.placeholder(tf.float32,[None,64,64,1],name="noisy_images")
  phase_train = tf.placeholder(tf.bool, [], name='phase_train')
  y_ = E.inference(x_, phase_train)
  
  init = tf.global_variables_initializer()
  E.sess.run(init)

  E.load_model(args.model_snapshot_file)

  with open(os.path.join(log_dir, 'log_file.pkl'), 'a') as outfile:
    logs = []
    for Ic, In, clean_image_name, noisy_image_name in tqdm(generator):
      In   = post_process(In)
      IReconstructed = denoise_image(In, stride)
      if args.save_images == 'True':
        # this will currently give an error
        Image.fromarray(IReconstructed.astype(numpy.uint8)).save(
          os.path.join(reconstructed_images_save_folder,
            os.path.basename(noisy_image_name).replace('noisy_image', 'recon_image'))
          )
      DATUM={}
      DATUM['file'] = os.path.basename(clean_image_name)
      DATUM['noise']=args.noise_level
      DATUM['noise_clean_psnr'] = measure.compare_psnr(im_true=Ic, im_test=In, dynamic_range=255.0)
      DATUM['recon_clean_psnr'] = measure.compare_psnr(im_true=Ic, im_test=IReconstructed, dynamic_range=255.0)
      logs.append(DATUM)
    pickle.dump(logs, outfile)
