# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

# This file is a modification of the following file: https://github.com/NVlabs/stylegan2/blob/master/projector.py.
# Adjustements made by Eklavya Sarkar (Idiap Research Institute, Biometrics Security and Privacy), Jan-Feb 2021,
#              -> Perceptual loss: mipgan_perceptual_loss()
#              -> Identity loss: mipgan_identity_loss()
#              -> ID-Difference loss: mipgan_id_diff_loss()
#              -> MS-SSIM loss: mipgan_ms_ssim_loss()
#              -> Other misc initialization and variable changes required for this implementation.

# For the purpose of generating morphing attacks as described in the paper:

# Sarkar, E., Korshunov, P., Colbois, L. and Marcel, S., Are GAN-based Morphs Threatening Face Recognition?, 2022.
# International Conference on Acoustics, Speech, & Signal Processing (ICASSP).

# The implementation is a modified version of the one described in:

# Zhang, H., Venkatesh, S., Ramachandra, R., Raja, K., Damer, N. and Busch, C., 2021. 
# Mipgan—generating strong and high quality morphing attacks using identity prior driven gan. 
# IEEE Transactions on Biometrics, Behavior, and Identity Science, 3(3), pp.365-383.
# URL: https://arxiv.org/abs/2009.01729

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.engine import Model
from modules import misc
import cv2
import h5py
import os

#----------------------------------------------------------------------------


class Morpher(object):
    def __init__(self, alpha):
        self.num_steps                  = 150
        self.dlatent_avg_samples        = 10000
        self.initial_learning_rate      = 0.03
        self.initial_noise_factor       = 0.05
        self.lr_rampdown_length         = 0.25
        self.lr_rampup_length           = 0.05
        self.noise_ramp_length          = 0.75
        self.regularize_noise_weight    = 1e5
        self.verbose                    = False
        self.clone_net                  = True

        self._Gs                    = None
        self._minibatch_size        = None
        self._dlatent_avg           = None
        self._dlatent_std           = None
        self._noise_vars            = None
        self._noise_init_op         = None
        self._noise_normalize_op    = None
        self._dlatents_var          = None
        self._noise_in              = None
        self._dlatents_expr         = None
        self._images_expr           = None
        self._target_images_var     = None
        self._lpips                 = None
        self._dist                  = None
        self._loss                  = None
        self._reg_sizes             = None
        self._lrate_in              = None
        self._opt                   = None
        self._opt_step              = None
        self._cur_step              = None

        self._dlatent_rand          = None
        self._dlatent_avg_tf        = None
        self._dlatent_interp_tf     = None
        self.dlatent_interp_tf      = None
        self._dlatent_rand_tf       = None
        self._image_rand_tf         = None
        self._feature_layer         = 'flatten_1'
        self._model                 = 'resnet50'
        self._x_1                   = None
        self._x_2                   = None
        self._B                     = None

        self._alpha                 = alpha
        self.proc_images_expr       = None
        self.proc_images_interp     = None
        
        self._B_proc_i_m_v2         = None
        self.i_m                    = None
        self._x_1_proc_v2           = None
        self._x_2_proc_v2           = None

        self.proc_i_m               = None
        self._B_x_1_proc_v2         = None
        self._B_x_2_proc_v2         = None

        self._loss_perceptual       = None
        self._loss_identity         = None
        self._loss_id_diff          = None
        self._loss_ms_ssim          = None

    def _info(self, *args):
        if self.verbose:
            print(*args)

    def set_network(self, Gs, minibatch_size=1):

        assert minibatch_size == 1
        self._Gs = Gs
        self._minibatch_size = minibatch_size
        if self._Gs is None:
            return
        if self.clone_net:
            self._Gs = self._Gs.clone()

        # Find dlatent stats.
        self._info('Finding W midpoint and stddev using %d samples...' % self.dlatent_avg_samples)
        latent_samples = np.random.RandomState(123).randn(self.dlatent_avg_samples, *self._Gs.input_shapes[0][1:])   # (10000,512)
        dlatent_samples = self._Gs.components.mapping.run(latent_samples, None)[:, :1, :]                            # [N, 1, 512] # N = 10000
        image_rand, self._dlatent_rand = self._Gs.run(latent_samples[1][None,:], None, return_dlatents=True)
        self._dlatent_rand = self._dlatent_rand[:, :1, :]                                                            # [1, 1, 512]
        self._dlatent_avg = np.mean(dlatent_samples, axis=0, keepdims=True)                                          # [1, 1, 512]
        self._dlatent_std = (np.sum((dlatent_samples - self._dlatent_avg) ** 2) / self.dlatent_avg_samples) ** 0.5
        self._info('std = %g' % self._dlatent_std)

        # Find noise inputs.
        self._info('Setting up noise inputs...')
        self._noise_vars = []
        noise_init_ops = []
        noise_normalize_ops = []
        while True:
            n = 'G_synthesis/noise%d' % len(self._noise_vars)
            if not n in self._Gs.vars:
                break
            v = self._Gs.vars[n]
            self._noise_vars.append(v)
            noise_init_ops.append(tf.assign(v, tf.random_normal(tf.shape(v), dtype=tf.float32)))
            noise_mean = tf.reduce_mean(v)
            noise_std = tf.reduce_mean((v - noise_mean)**2)**0.5
            noise_normalize_ops.append(tf.assign(v, (v - noise_mean) / noise_std))
            self._info(n, v)
        self._noise_init_op = tf.group(*noise_init_ops)
        self._noise_normalize_op = tf.group(*noise_normalize_ops)

        # Image output graph
        self._info('Building image output graph...')
        self._dlatents_var  = tf.Variable(tf.zeros([self._minibatch_size] + list(self._dlatent_avg.shape[1:])), name='dlatents_var') # (1,1,512)
        self._noise_in      = tf.placeholder(tf.float32, [], name='noise_in')                                                        # () 

        # Set VGGFace biometric network and extract embeddings
        self._B = self.return_vggface_network()

        # Initialize
        self.dlatent_interp_tf   = tf.Variable(tf.zeros([1, 1, 512]), name='dlatent_interp_tf')                           # (1, 1, 512)
        self.dlatent_interp_expr = tf.tile(self.dlatent_interp_tf, [1, self._Gs.components.synthesis.input_shape[1], 1])  # (1, 18, 512)
        self.i_m = self._Gs.components.synthesis.get_output_for(self.dlatent_interp_expr, randomize_noise=False)          # (1, 3, 1024, 1024)
        
        # Loss graph
        self._info('Building loss graph...')
        self._x_1 = tf.Variable(tf.zeros([self._minibatch_size, 256, 256, 3]), name='x_1') # (1, 256, 256, 3)
        self._x_2 = tf.Variable(tf.zeros([self._minibatch_size, 256, 256, 3]), name='x_2') # (1, 256, 256, 3)

        # Preprocess input images
        self._x_1_proc_v2 = self.preprocess_input(self._x_1, version=2)
        self._x_2_proc_v2 = self.preprocess_input(self._x_2, version=2)

        # Extract Embeddings
        self._B_x_1_proc_v2 = self._B(self._x_1_proc_v2)
        self._B_x_2_proc_v2 = self._B(self._x_2_proc_v2)
        
        # Downsize, preprocess, extract
        self.proc_i_m       = self.downsize_img_tf(self.i_m)                   # (1, 256, 256, 3)
        self.proc_i_m_v2    = self.preprocess_input(self.proc_i_m, version=2)  # (1, 256, 256, 3)
        self._B_proc_i_m_v2 = self._B(self.proc_i_m_v2)                        # (1, 2048)

        # Choose loss
        self._loss = self.mipgan_loss(self._B_x_1_proc_v2,
                                      self._B_x_2_proc_v2,
                                      self.proc_i_m,
                                      self._B_proc_i_m_v2,
                                      self._x_1,
                                      self._x_2)

        # Noise regularization graph.
        self._info('Building noise regularization graph...')
        reg_loss = 0.0
        for v in self._noise_vars:
            sz = v.shape[2]
            while True:
                reg_loss += tf.reduce_mean(v * tf.roll(v, shift=1, axis=3))**2 + tf.reduce_mean(v * tf.roll(v, shift=1, axis=2))**2
                if sz <= 8:
                    break # Small enough already
                v = tf.reshape(v, [1, 1, sz//2, 2, sz//2, 2]) # Downscale
                v = tf.reduce_mean(v, axis=[3, 5])
                sz = sz // 2
        self._loss += reg_loss * self.regularize_noise_weight

        # Optimizer.
        self._info('Setting up optimizer...')
        self._lrate_in = tf.placeholder(tf.float32, [], name='lrate_in')
        self._opt = dnnlib.tflib.Optimizer(learning_rate=self._lrate_in)
        self._opt.register_gradients(self._loss, [self.dlatent_interp_tf] + self._noise_vars)
        self._opt_step = self._opt.apply_updates()

    def mipgan_perceptual_loss(self, proc_i_m, x_1, x_2):
        '''
        Calculates and returns the scalar of MIPGAN's first loss: the perceptual loss.
        '''
        # Preprocess (version 1 for VGG16)
        proc_i_m_v1 = self.preprocess_input(proc_i_m,  version=1) # (1, 256, 256, 3)
        proc_x_1    = self.preprocess_input(x_1, version=1)       # (1, 256, 256, 3)
        proc_x_2    = self.preprocess_input(x_2, version=1)       # (1, 256, 256, 3)
        # Variables
        selected_layers = ['conv1_1', 'conv1_2', 'conv2_2', 'conv3_3']
        vgg_16 = VGGFace(model='vgg16', include_top=False, input_shape=(256, 256, 3), pooling='avg')
        sum_1  = 0
        sum_2  = 0
        # Loop through the layers
        for layer in selected_layers:
            # Make get layer in question
            out = vgg_16.get_layer(layer).output
            vgg_16_model = Model(vgg_16.input, out)
            # Get embeddings
            vgg_embed_i_m = vgg_16_model(proc_i_m_v1)
            vgg_embed_x_1 = vgg_16_model(proc_x_1)
            vgg_embed_x_2 = vgg_16_model(proc_x_2)
            # Differences
            vgg_embed_diff_1 = tf.math.subtract(vgg_embed_x_1, vgg_embed_i_m, name='vgg_embeds_diff_1')
            vgg_embed_diff_2 = tf.math.subtract(vgg_embed_x_2, vgg_embed_i_m, name='vgg_embeds_diff_2')
            # Squared L2-Norms
            vgg_sqrt_l2_norms_1 = tf.math.sqrt(tf.norm(vgg_embed_diff_1, ord=2), name='squared_l2_norm_1')
            vgg_sqrt_l2_norms_2 = tf.math.sqrt(tf.norm(vgg_embed_diff_2, ord=2), name='squared_l2_norm_2')
            # Divisions
            ratio = 1 / vgg_embed_i_m.get_shape().as_list()[-1] # Check if this actually the "number of features in layer i" value
            # Update
            sum_1 += tf.math.multiply(vgg_sqrt_l2_norms_1, ratio)
            sum_2 += tf.math.multiply(vgg_sqrt_l2_norms_2, ratio)
        # Take half of both sums
        loss = ((1-self._alpha) * sum_1) + (self._alpha * sum_2)
        # Return
        return loss

    def get_cosine_distances(self, B_x_1_norm, B_x_2_norm, B_i_m_norm):
        '''
        Calculates and returns the two cosine distances, both required for the id_diff and identity losses.
        '''
        # First term
        numerator_first_term       = tf.reduce_sum(tf.math.multiply(B_x_1_norm, B_i_m_norm), name='numerator_first_term')
        denominator_first_term     = tf.math.multiply(tf.norm(B_x_1_norm, ord=2), tf.norm(B_i_m_norm, ord=2), name='denominator_first_term')
        cos_similarity_first_term  = tf.math.divide(numerator_first_term, denominator_first_term)
        cosine_dist_1              = 1 - cos_similarity_first_term
        # Second term
        numerator_second_term      = tf.reduce_sum(tf.math.multiply(B_x_2_norm, B_i_m_norm, name='numerator_second_term'))
        denominator_second_term    = tf.math.multiply(tf.norm(B_x_2_norm, ord=2), tf.norm(B_i_m_norm, ord=2), name='denominator_second_term')
        cos_similarity_second_term = tf.math.divide(numerator_second_term, denominator_second_term)
        cosine_dist_2              = 1 - cos_similarity_second_term
        return cosine_dist_1, cosine_dist_2

    def mipgan_identity_loss(self, cosine_dist_1, cosine_dist_2):
        '''
        Calculates and returns the scalar of MIPGAN's second loss: the identity loss.
        '''
        # Sum & Division
        weighted_cosine_dist_1 = cosine_dist_1 * (1-self._alpha)
        weighted_cosine_dist_2 = cosine_dist_2 * self._alpha
        return tf.math.add_n([weighted_cosine_dist_1, weighted_cosine_dist_2], name='sum_cos_dists')

    def mipgan_id_diff_loss(self, cosine_dist_1, cosine_dist_2):
        '''
        Calculates and returns the scalar of MIPGAN's third loss: the id_diff loss.
        '''
        # Substraction & L1 norm
        weighted_cosine_dist_1 = cosine_dist_1 * (1-self._alpha)
        weighted_cosine_dist_2 = cosine_dist_2 * self._alpha
        cosine_dist_diff = tf.math.subtract(weighted_cosine_dist_1, weighted_cosine_dist_2, name='substract_cos_dist')
        return tf.norm(cosine_dist_diff, ord=1)

    def mipgan_ms_ssim_loss(self, proc_i_m, x_1, x_2):
        '''
        Calculates and returns the scalar of MIPGAN's fourth loss: the ms_ssim loss.
        Remember this is a value one has to maximize to optimize, not minimize.
        '''
        loss_1 = tf.image.ssim_multiscale(x_1, proc_i_m, max_val=255)
        loss_2 = tf.image.ssim_multiscale(x_2, proc_i_m, max_val=255)
        return - ((loss_1*(1-self._alpha)) + (loss_2*self._alpha))

    def mipgan_loss(self, B_x_1_proc_v2, B_x_2_proc_v2, proc_i_m, B_proc_i_m_v2, x_1, x_2, 
                    lambda_perceptual=0.0002, lambda_identity=10, lambda_id_diff=1, lambda_ms_ssim=1):
        '''
        Adapted equation from paper 'MIPGAN - Generating Robust and High Quality Morph Attacks Using Identity Prior Driven GAN'
        '''
        # L2 Normalize vectors, so their length (norm) is 1
        B_x_1_norm = tf.nn.l2_normalize(B_x_1_proc_v2, name='norm_B_x_1') # tf.norm(B_x_1_norm)=1
        B_x_2_norm = tf.nn.l2_normalize(B_x_2_proc_v2, name='norm_B_x_2') # tf.norm(B_x_2_norm)=1
        B_i_m_norm = tf.nn.l2_normalize(B_proc_i_m_v2, name='norm_B_i_m') # tf.norm(B_i_m_norm)=1
        # Get cosine distances
        cosine_dist_1, cosine_dist_2 = self.get_cosine_distances(B_x_1_norm, B_x_2_norm, B_i_m_norm)
        # Get losses
        self._loss_perceptual = self.mipgan_perceptual_loss(proc_i_m, x_1, x_2)         * lambda_perceptual
        self._loss_identity   = self.mipgan_identity_loss(cosine_dist_1, cosine_dist_2) * lambda_identity
        self._loss_id_diff    = self.mipgan_id_diff_loss(cosine_dist_1, cosine_dist_2)  * lambda_id_diff
        self._loss_ms_ssim    = self.mipgan_ms_ssim_loss(proc_i_m, x_1, x_2)            * lambda_ms_ssim
        # Final loss
        return self._loss_perceptual + self._loss_identity + self._loss_id_diff + self._loss_ms_ssim

    def get_middle_morph_latent(self, latents_dir, m_name):

        '''
        Reads the latent vectors of projected images, and creates the middle interpolation morph,
        sends it through the Generator to get the corresponding (1, 3, 1024, 1024) image,
        before processing it to a (1, 256, 256, 3) image (channels last).
        '''
        # Get morph file names
        f1, f2 = m_name
        suffix = '.hdf5'
        f1 = f1 + suffix # Modify according to hdf5 file names
        f2 = f2 + suffix # Modify according to hdf5 file names

        # Read existing latent vectors of each input image (previously saved as hdf5)
        latents_path = [os.path.join(latents_dir, f1), os.path.join(latents_dir, f2)]
        
        # Convert hdf5 file to numpy
        latent_x_1 = np.array(h5py.File(latents_path[0], 'r')['array'])[None, None, :] # (1, 1, 512)
        latent_x_2 = np.array(h5py.File(latents_path[1], 'r')['array'])[None, None, :] # (1, 1, 512)

        # Get the middle interpolation: 0.5*x1 + (1-0.5)*x2
        scaled_x_1 = 0.5 * latent_x_1
        scaled_x_2 = 0.5 * latent_x_2
        _dlatent_interp = scaled_x_1 + scaled_x_2

        # Return
        return _dlatent_interp # (1, 1, 512)

    def downsize_img_tf(self, images_expr, size=256):
        '''
        Downsizes the tensor of an image, and puts the channel last.
        @param _images_expr: an image tensor with the channels first, eg: (1, 3, 1024, 1024)
        '''
        # Downsize
        proc_images_expr = (images_expr + 1) * (255 / 2) # (1, 3, 1024, 1024)
        sh = proc_images_expr.shape.as_list()
        if sh[2] > size:
            factor = sh[2] // size
            proc_images_expr = tf.reduce_mean(tf.reshape(proc_images_expr, [-1, sh[1], sh[2] // factor , factor, sh[2] // factor, factor]), axis=[3,5])
        # Channels last
        sh = proc_images_expr.shape.as_list()
        proc_images_expr = tf.transpose(proc_images_expr, (0, 2, 3, 1), name='channels_last') # (1, 256, 256, 3)
        # Return
        return proc_images_expr

    def return_vggface_network(self):
        '''
        Creates and returns a Keras VGGFace network.
        '''
        # Biometric Network B
        # B_VGG = VGGFace(model=self._model)
        # out = B_VGG.get_layer(self._feature_layer).output
        # return Model(B_VGG.input, out)
        return VGGFace(model=self._model, include_top=False, input_shape=(256, 256, 3), pooling='avg')

    def _preprocess_numpy_input(self, x, version, data_format, mode, **kwargs):
        """Preprocesses a Numpy array encoding a batch of images.
        # Arguments
            x: Input array, 3D or 4D.
            data_format: Data format of the image array.
            mode: One of "caffe", "tf" or "torch".
                - caffe: will convert the images from RGB to BGR,
                    then will zero-center each color channel with
                    respect to the ImageNet dataset,
                    without scaling.
                - tf: will scale pixels between -1 and 1,
                    sample-wise.
                - torch: will scale pixels between 0 and 1 and then
                    will normalize each channel with respect to the
                    ImageNet dataset.
        # Returns
            Preprocessed Numpy array.
        """
        backend = tf.keras.backend
        if not issubclass(x.dtype.type, np.floating):
            x = x.astype(backend.floatx(), copy=False)

        if mode == 'tf':
            x /= 127.5
            x -= 1.
            return x

        if mode == 'torch':
            x /= 255.
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            if data_format == 'channels_first':
                # 'RGB'->'BGR'
                if x.ndim == 3:
                    x = x[::-1, ...]
                else:
                    x = x[:, ::-1, ...]
            else:
                # 'RGB'->'BGR'
                x = x[..., ::-1]
            # mean = [103.939, 116.779, 123.68]
            if version == 1:
                mean = [93.5940, 104.7624, 129.1863]
            elif version == 2:
                mean = [91.4953, 103.8827, 131.0912]
            std = None

        # Zero-center by mean pixel
        if data_format == 'channels_first':
            if x.ndim == 3:
                x[0, :, :] -= mean[0]
                x[1, :, :] -= mean[1]
                x[2, :, :] -= mean[2]
                if std is not None:
                    x[0, :, :] /= std[0]
                    x[1, :, :] /= std[1]
                    x[2, :, :] /= std[2]
            else:
                x[:, 0, :, :] -= mean[0]
                x[:, 1, :, :] -= mean[1]
                x[:, 2, :, :] -= mean[2]
                if std is not None:
                    x[:, 0, :, :] /= std[0]
                    x[:, 1, :, :] /= std[1]
                    x[:, 2, :, :] /= std[2]
        else:
            x[..., 0] -= mean[0]
            x[..., 1] -= mean[1]
            x[..., 2] -= mean[2]
            if std is not None:
                x[..., 0] /= std[0]
                x[..., 1] /= std[1]
                x[..., 2] /= std[2]
        return x

    def get_submodules_from_kwargs(self, kwargs):
        backend = kwargs.get('backend', _KERAS_BACKEND)
        layers  = kwargs.get('layers', _KERAS_LAYERS)
        models  = kwargs.get('models', _KERAS_MODELS)
        utils   = kwargs.get('utils', _KERAS_UTILS)
        for key in kwargs.keys():
            if key not in ['backend', 'layers', 'models', 'utils']:
                raise TypeError('Invalid keyword argument: %s', key)
        return backend, layers, models, utils

    def _preprocess_symbolic_input(self, x, version, data_format, mode, **kwargs):
        """Preprocesses a tensor encoding a batch of images.
        # Arguments
            x: Input tensor, 3D or 4D.
            data_format: Data format of the image tensor.
            mode: One of "caffe", "tf" or "torch".
                - caffe: will convert the images from RGB to BGR,
                    then will zero-center each color channel with
                    respect to the ImageNet dataset,
                    without scaling.
                - tf: will scale pixels between -1 and 1,
                    sample-wise.
                - torch: will scale pixels between 0 and 1 and then
                    will normalize each channel with respect to the
                    ImageNet dataset.
        # Returns
            Preprocessed tensor.
        """
        _IMAGENET_MEAN = None

        #backend, _, _, _ = self.get_submodules_from_kwargs(kwargs)
        backend = tf.keras.backend

        if mode == 'tf':
            x /= 127.5
            x -= 1.
            return x

        if mode == 'torch':
            x /= 255.
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            if data_format == 'channels_first':
                # 'RGB'->'BGR'
                if backend.ndim(x) == 3:
                    x = x[::-1, ...]
                else:
                    x = x[:, ::-1, ...]
            else:
                # 'RGB'->'BGR'
                x = x[..., ::-1]
            #mean = [103.939, 116.779, 123.68]
            if version == 1:
                mean = [93.5940, 104.7624, 129.1863]
            elif version == 2:
                mean = [91.4953, 103.8827, 131.0912]
            std = None

        if _IMAGENET_MEAN is None:
            _IMAGENET_MEAN = backend.constant(-np.array(mean))

        # Zero-center by mean pixel
        if backend.dtype(x) != backend.dtype(_IMAGENET_MEAN):
            x = backend.bias_add(
                x, backend.cast(_IMAGENET_MEAN, backend.dtype(x)),
                data_format=data_format)
        else:
            x = backend.bias_add(x, _IMAGENET_MEAN, data_format)
        if std is not None:
            x /= std
        return x

    def preprocess_input(self, x, version, data_format=None, mode='caffe', **kwargs):
        """Preprocesses a tensor or Numpy array encoding a batch of images.
        # Arguments
            x: Input Numpy or symbolic tensor, 3D or 4D.
                The preprocessed data is written over the input data
                if the data types are compatible. To avoid this
                behaviour, `numpy.copy(x)` can be used.
            data_format: Data format of the image tensor/array.
            mode: One of "caffe", "tf" or "torch".
                - caffe: will convert the images from RGB to BGR,
                    then will zero-center each color channel with
                    respect to the ImageNet dataset,
                    without scaling.
                - tf: will scale pixels between -1 and 1,
                    sample-wise.
                - torch: will scale pixels between 0 and 1 and then
                    will normalize each channel with respect to the
                    ImageNet dataset.
        # Returns
            Preprocessed tensor or Numpy array.
        # Raises
            ValueError: In case of unknown `data_format` argument.
        """
        #backend, _, _, _ = self.get_submodules_from_kwargs(kwargs)
        backend = tf.keras.backend

        if data_format is None:
            data_format = backend.image_data_format()
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format ' + str(data_format))

        if isinstance(x, np.ndarray):
            return self._preprocess_numpy_input(x, version, data_format=data_format, mode=mode, **kwargs)
        else:
            return self._preprocess_symbolic_input(x, version, data_format=data_format, mode=mode, **kwargs)

    def preprocess_input_tf(self, x, data_format=None, version=1):
        # x_temp = tf.identity(x)
        x_temp = x
        if data_format is None:
            data_format = tf.keras.backend.image_data_format()
        assert data_format in {'channels_last', 'channels_first'}

        if version == 1:
            if data_format == 'channels_first':
                x_temp = x_temp[:, ::-1, ...]
                x_temp[:, 0, :, :] -= 93.5940
                x_temp[:, 1, :, :] -= 104.7624
                x_temp[:, 2, :, :] -= 129.1863
            else:
                x_temp = x_temp[..., ::-1]
                x_temp[:, :, :, 0] -= 93.5940
                x_temp[:, :, :, 1] -= 104.7624
                x_temp[:, :, :, 2] -= 129.1863

        elif version == 2:
            if data_format == 'channels_first':
                x_temp = x_temp[:, ::-1, ...]
                x_temp[:, 0, :, :] -= 91.4953
                x_temp[:, 1, :, :] -= 103.8827
                x_temp[:, 2, :, :] -= 131.0912
            else:
                tf_var = tf.Variable(tf.zeros(shape=x_temp.shape))
                x_temp = x_temp[..., ::-1]
                tf_var[:, :, :, 0] = x_temp[:, :, :, 0] - 91.4953
                tf_var[:, :, :, 1] = x_temp[:, :, :, 1] - 103.8827
                tf_var[:, :, :, 2] = x_temp[:, :, :, 2] - 131.0912

                x_temp[:, :, :, 0] -= 91.4953
                x_temp[:, :, :, 1] -= 103.8827
                x_temp[:, :, :, 2] -= 131.0912
        else:
            raise NotImplementedError

        return tf_var

    def restore_image(self, x, data_format=None):
        x_temp = np.copy(x)
        # if data_format is None:
        data_format = tf.keras.backend.image_data_format()
        assert data_format in {'channels_last', 'channels_first'}
        mean = [93.5940, 104.7624, 129.1863]

        # Zero-center by mean pixel
        if data_format == 'channels_first':
            if x_temp.ndim == 3:
                x_temp[0, :, :] += mean[0]
                x_temp[1, :, :] += mean[1]
                x_temp[2, :, :] += mean[2]
            else:
                x_temp[:, 0, :, :] += mean[0]
                x_temp[:, 1, :, :] += mean[1]
                x_temp[:, 2, :, :] += mean[2]
        else:
            x_temp[..., 0] += mean[0]
            x_temp[..., 1] += mean[1]
            x_temp[..., 2] += mean[2]

        if data_format == 'channels_first':
            # 'BGR'->'RGB'
            if x_temp.ndim == 3:
                x_temp = x_temp[::-1, ...]
            else:
                x_temp = x_temp[:, ::-1, ...]
        else:
            # 'BGR'->'RGB'
            x_temp = x_temp[..., ::-1]
        # Return
        return x_temp

    def run(self, target_images):
        # Run to completion.
        self.start(target_images)
        while self._cur_step < self.num_steps:
            self.step()

        # Collect results.
        pres = dnnlib.EasyDict()

        # For Perceptual or (Euclidean) Biometric loss
        # pres.dlatents = self.get_dlatents()
        # pres.noises = self.get_noises()
        # pres.images = self.get_images()

        # For MIPGAN loss
        pres.dlatents = self.get_dlatent_interp()
        pres.noises = self.get_noises()
        pres.images = self.get_images_interp()

        return pres

    def start(self, target_images, latents_dir, m_name):
        assert self._Gs is not None

        # Prepare target images.
        self._info('Preparing target images...')
        target_images = np.asarray(target_images, dtype='float32')
        target_images = (target_images + 1) * (255 / 2)
        sh = target_images.shape # (2, 3, 1024, 1024)
        
        if sh[2] > self._x_1.shape[2]:
            factor = sh[2] // self._x_1.shape[2]
            target_images = np.reshape(target_images, [-1, sh[1], sh[2] // factor, factor, sh[3] // factor, factor]).mean((3, 5))
        sh = target_images.shape # (2, 3, 256, 256)
        target_images = np.transpose(target_images, (0, 2, 3, 1)) # (2, 256, 256, 3)

        # Initialize optimization state.
        self._info('Initializing optimization state...')
        tflib.set_vars({self._x_1: target_images[0][None, :, :, :], # (1, 256, 256, 3)
                        self._x_2: target_images[1][None, :, :, :], # (1, 256, 256, 3)
                        self.dlatent_interp_tf: self.get_middle_morph_latent(latents_dir, m_name)
                        })
        tflib.run(self._noise_init_op)
        self._opt.reset_optimizer_state()
        self._cur_step = 0

    def step(self):
        assert self._cur_step is not None
        if self._cur_step >= self.num_steps:
            return

        # Hyperparameters.
        t = self._cur_step / self.num_steps
        noise_strength = self._dlatent_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / self.lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / self.lr_rampup_length)
        learning_rate = self.initial_learning_rate * lr_ramp

        # Train.
        feed_dict = {self._noise_in: noise_strength, self._lrate_in: learning_rate}
        run_list = [self._opt_step, self._loss_perceptual, self._loss_identity, self._loss_id_diff, self._loss_ms_ssim, self._loss]
        _, loss_perceptual_value, loss_identity_value, loss_id_diff_value, loss_ms_ssim_value, loss_value = tflib.run(run_list, feed_dict)        
        tflib.run(self._noise_normalize_op)

        # Print status.
        self._cur_step += 1
        #if self._cur_step == self.num_steps or self._cur_step % 10 == 0:
        self._info('%-8d%-12g%-12g%-12g%-12g%-12g' % (self._cur_step, loss_perceptual_value, loss_identity_value, loss_id_diff_value, loss_ms_ssim_value, loss_value))
        if self._cur_step == self.num_steps:
            self._info('Done.')

    def get_cur_step(self):
        return self._cur_step

    def get_dlatents(self):
        return tflib.run(self._dlatents_expr, {self._noise_in: 0}) # (1,18,512)

    def get_noises(self):
        return tflib.run(self._noise_vars)

    def get_images(self):
        return tflib.run(self._images_expr, {self._noise_in: 0}) # (1,3,1024,1024)
    #
    def get_dlatent_interp(self):
        return tflib.run(self._dlatent_interp_tf) # (1,18,512)

    def get_images_interp(self):
        return tflib.run(self.i_m, {self._noise_in: 0}) # (1,3,1024,1024)

#----------------------------------------------------------------------------
