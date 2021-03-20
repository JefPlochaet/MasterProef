import os.path
import shutil
import time
import numpy as np
#import tensorflow as tf --aangepast naar volgende lijn (update van de package)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
import random
from nets import models_factory
from data_provider import datasets_factory
from utils import preprocess
from utils import metrics
#from skimage.measure import compare_ssim --aangepast naar volgende lijn (update van de package)
from skimage.metrics import structural_similarity as compare_ssim

# -----------------------------------------------------------------------------
#-----------------------------------------------
#Oplossing voor error veroorzaakt door xrange (zonder code aan te passen)
try:
    # Python 2
    xrange
except NameError:
    # Python 3, xrange is now named range
    xrange = range
#-----------------------------------------------
FLAGS = tf.app.flags.FLAGS

# data I/O
tf.app.flags.DEFINE_string('dataset_name', 'auto',
                           'The name of dataset.')
if FLAGS.dataset_name == 'auto':
    tf.app.flags.DEFINE_string('train_data_paths',
                           '/workspace/data/auto-train-frontside.npz',
                           'train data paths.')
    tf.app.flags.DEFINE_string('valid_data_paths',
                           '/workspace/data/auto-validatie-frontside.npz',
                           'validation data paths.')
    tf.app.flags.DEFINE_string('test_data_paths',
                           '/workspace/data/auto-test-frontside-l4.npz',
                           'test data paths.')
else:
    tf.app.flags.DEFINE_string('train_data_paths',
                           'data/moving-mnist-train.npz',
                           'train data paths.')
    tf.app.flags.DEFINE_string('valid_data_paths',
                           'data/moving-mnist-valid.npz',
                           'validation data paths.')
tf.app.flags.DEFINE_string('save_dir', '/workspace/checkpoints-frontside',
                            'dir to store trained net.')
tf.app.flags.DEFINE_string('gen_frm_dir', '/workspace/results-frontside',
                           'dir to store result.')
# model
tf.app.flags.DEFINE_string('model_name', 'predrnn_pp',
                           'The name of the architecture.')
tf.app.flags.DEFINE_string('pretrained_model', '/workspace/checkpoints-frontside/model.ckpt-60000',
                           'file of a pretrained model to initialize from.')
tf.app.flags.DEFINE_integer('input_length', 3,
                            'encoder hidden states.')
tf.app.flags.DEFINE_integer('seq_length', 4,
                            'total input and output length.')
tf.app.flags.DEFINE_integer('img_width', 240,
                            'input image width.')
tf.app.flags.DEFINE_integer('img_height', 164,
                            'input image height.')
tf.app.flags.DEFINE_integer('img_channel', 1,
                            'number of image channel.')
tf.app.flags.DEFINE_integer('stride', 1,
                            'stride of a convlstm layer.')
tf.app.flags.DEFINE_integer('filter_size', 5,
                            'filter of a convlstm layer.')
tf.app.flags.DEFINE_string('num_hidden', '128,64,64,64',
                           'COMMA separated number of units in a convlstm layer.')
tf.app.flags.DEFINE_integer('patch_size', 4,
                            'patch size on one dimension.') #origineel 4
tf.app.flags.DEFINE_boolean('layer_norm', True,
                            'whether to apply tensor layer norm.')
# optimization
tf.app.flags.DEFINE_float('lr', 0.001,
                          'base learning rate.')
tf.app.flags.DEFINE_boolean('reverse_input', False,
                            'whether to reverse the input frames while training.') #origineel True
tf.app.flags.DEFINE_integer('batch_size', 8,
                            'batch size for training.')
tf.app.flags.DEFINE_integer('max_iterations', 60000,
                            'max num of steps.') #origineel 80000
tf.app.flags.DEFINE_integer('display_interval', 1,
                            'number of iters showing training loss.')
tf.app.flags.DEFINE_integer('test_interval', 15000,
                            'number of iters for test.') #origineel 2000
tf.app.flags.DEFINE_integer('snapshot_interval', 60000,
                            'number of iters saving models.') #origineel 10000

class Model(object):
    def __init__(self):
        # inputs
        self.x = tf.placeholder(tf.float32,
                                [FLAGS.batch_size,
                                 FLAGS.seq_length,
                                 int(FLAGS.img_height/FLAGS.patch_size),
                                 int(FLAGS.img_width/FLAGS.patch_size),
                                 int(FLAGS.patch_size*FLAGS.patch_size*FLAGS.img_channel)]) #int moeten toevoegen, anders werden dit floats?

        self.mask_true = tf.placeholder(tf.float32,
                                        [FLAGS.batch_size,
                                         FLAGS.seq_length-FLAGS.input_length-1,
                                         int(FLAGS.img_height/FLAGS.patch_size),
                                         int(FLAGS.img_width/FLAGS.patch_size),
                                         int(FLAGS.patch_size*FLAGS.patch_size*FLAGS.img_channel)])

        grads = []
        loss_train = []
        self.pred_seq = []
        self.tf_lr = tf.placeholder(tf.float32, shape=[])
        num_hidden = [int(x) for x in FLAGS.num_hidden.split(',')]
        print(num_hidden)
        num_layers = len(num_hidden)
        with tf.variable_scope(tf.get_variable_scope()):
            # define a model
            output_list = models_factory.construct_model(
                FLAGS.model_name, self.x,
                self.mask_true,
                num_layers, num_hidden,
                FLAGS.filter_size, FLAGS.stride,
                FLAGS.seq_length, FLAGS.input_length,
                FLAGS.layer_norm)
            gen_ims = output_list[0]
            loss = output_list[1]
            pred_ims = gen_ims[:,FLAGS.input_length-1:]
            self.loss_train = loss / FLAGS.batch_size
            # gradients
            all_params = tf.trainable_variables()
            grads.append(tf.gradients(loss, all_params))
            self.pred_seq.append(pred_ims)

        self.train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

        # session
        variables = tf.global_variables()
        self.saver = tf.train.Saver(variables)
        init = tf.global_variables_initializer()
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config = configProt)
        self.sess.run(init)
        if FLAGS.pretrained_model:
            self.saver.restore(self.sess, FLAGS.pretrained_model)

    def train(self, inputs, lr, mask_true):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.tf_lr: lr})
        feed_dict.update({self.mask_true: mask_true})
        loss, _ = self.sess.run((self.loss_train, self.train_op), feed_dict)
        return loss

    def test(self, inputs, mask_true):
        feed_dict = {self.x: inputs}
        feed_dict.update({self.mask_true: mask_true})
        gen_ims = self.sess.run(self.pred_seq, feed_dict)
        return gen_ims

    def save(self, itr):
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
        self.saver.save(self.sess, checkpoint_path, global_step=itr)
        print('saved to ' + FLAGS.save_dir)

def main(argv=None):

    tf.disable_eager_execution() #toegevoegd anders error

    # load data
    _, test_input_handle = datasets_factory.data_provider(
        FLAGS.dataset_name, FLAGS.train_data_paths, FLAGS.test_data_paths,
        FLAGS.batch_size, FLAGS.img_width)

    print("Initializing models")
    model = Model()
    lr = FLAGS.lr

    print('test...')
    test_input_handle.begin(do_shuffle = False)
    res_path = os.path.join(FLAGS.gen_frm_dir, 'test')
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse,ssim,psnr,fmae,sharp= [],[],[],[],[]
    for i in xrange(FLAGS.seq_length - FLAGS.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        fmae.append(0)
        sharp.append(0)
    mask_true = np.zeros((FLAGS.batch_size,
                            FLAGS.seq_length-FLAGS.input_length-1,
                            int(FLAGS.img_height/FLAGS.patch_size),
                            int(FLAGS.img_width/FLAGS.patch_size),
                            FLAGS.patch_size**2*FLAGS.img_channel))
    while(test_input_handle.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()
        test_dat = preprocess.reshape_patch(test_ims, FLAGS.patch_size)
        img_gen = model.test(test_dat, mask_true)

        # concat outputs of different gpus along batch
        img_gen = np.concatenate(img_gen)
        img_gen = preprocess.reshape_patch_back(img_gen, FLAGS.patch_size)
        # MSE per frame
        for i in xrange(FLAGS.seq_length - FLAGS.input_length):
            x = test_ims[:,i + FLAGS.input_length,:,:,0]
            gx = img_gen[:,i,:,:,0]
            fmae[i] += metrics.batch_mae_frame_float(gx, x)
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)
            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
            for b in xrange(FLAGS.batch_size):
                sharp[i] += np.max(
                    cv2.convertScaleAbs(cv2.Laplacian(pred_frm[b],3)))
                score, _ = compare_ssim(pred_frm[b],real_frm[b],full=True)
                ssim[i] += score

        # save prediction examples
        if batch_id <= 10:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            for i in xrange(FLAGS.seq_length):
                name = 'gt' + str(i+1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0,i,:,:,:] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in xrange(FLAGS.seq_length-FLAGS.input_length):
                name = 'pd' + str(i+1+FLAGS.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_gen[0,i,:,:,:]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
        test_input_handle.next()
    avg_mse = avg_mse / (batch_id*FLAGS.batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in xrange(FLAGS.seq_length - FLAGS.input_length):
        print(img_mse[i] / (batch_id*FLAGS.batch_size))
    psnr = np.asarray(psnr, dtype=np.float32)/batch_id
    fmae = np.asarray(fmae, dtype=np.float32)/batch_id
    ssim = np.asarray(ssim, dtype=np.float32)/(FLAGS.batch_size*batch_id)
    sharp = np.asarray(sharp, dtype=np.float32)/(FLAGS.batch_size*batch_id)
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in xrange(FLAGS.seq_length - FLAGS.input_length):
        print(psnr[i])
    print('fmae per frame: ' + str(np.mean(fmae)))
    for i in xrange(FLAGS.seq_length - FLAGS.input_length):
        print(fmae[i])
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in xrange(FLAGS.seq_length - FLAGS.input_length):
        print(ssim[i])
    print('sharpness per frame: ' + str(np.mean(sharp)))
    for i in xrange(FLAGS.seq_length - FLAGS.input_length):
        print(sharp[i])

if __name__ == '__main__':
    tf.app.run()
