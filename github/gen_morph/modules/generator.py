# Copyright (c) 2021, Idiap Research Institute. All rights reserved.
#
# This work is made available under a custom license, Non-Commercial Research and Educational Use Only 
# To view a copy of this license, visit
# https://gitlab.idiap.ch/bob/bob.paper.ijcb2021_synthetic_dataset/-/blob/master/LICENSE.txt


import pickle
import dnnlib
from dnnlib import tflib
import numpy as np
from bob.extension import rc
import utils

class StyleGAN2(object):
    def __init__(self, sg2_path=rc['sg2_morph.sg2_path'], randomize_noise=False):
        """
        Instanciate the StyleGAN2 generator network. Cf.
        T. Karras, *et al.*, \"Analyzing and improving the quality of StyleGAN\", in *Proc.
        IEEE/CVF Conference on Computer Vision and Patter Recognition,
        2020.*
        Repository : https://github.com/NVlabs/stylegan2
        
        :param sg2_path: Path to the pickled pretrained network
        :param randomize_noise: Whether to randomly sample the noise inputs of the synthetizer, or fix them once 
                                and for all at initialization. 
        """
        with open(sg2_path, 'rb') as pkl_file:
            _G, _D, Gs = pickle.load(pkl_file)
        self.network = Gs
        self.latent_dim = Gs.input_shape[-1]
        self.run_kwargs = {'randomize_noise': randomize_noise}

        if not randomize_noise:
            noise_vars = [var for name, var in self.network.components.synthesis.vars.items() if name.startswith('noise')]
            tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
    
    def run_from_Z(self, latents, truncation_psi=0.5, return_w_latents=False, **kwargs):
        """
        Run the generator from the input latent space Z

        Inputs:
        :param latents: batch of latent vectors, of the shape [batch_size, latent_dim].
        :param truncation_psi: (float in [0,1]) value of psi when applying the truncation trick (cf. T. Karras, *et al.*, \"A style-based
                                generator architecture for generative adversarial networks\", in *Proc. CVPR*, 2018).
                                A value of 0 applies complete truncation (meaning one can only generate the mean face), 
                                while a value of 1 applies no truncation.
        :param return_w_latents: whether to return the W-space latents as additional output
        :param **kwargs: other parameters that will be feeded to the Network.run method

        Outputs:

        :return images: Batch of generated images in bob format : tensor with shape [batch_size, 3, 1024, 1024] of uint8
                        values in the range [0, 255]
        :return w_latents: Batch of W-space latents vector, of the shape [batch_size, latent_dim]. 
                            Only returned if `return_w_latents` is True.
        """
        run_kwargs = self.run_kwargs
        run_kwargs['truncation_psi'] = truncation_psi
        run_kwargs['return_dlatents'] = return_w_latents
        run_kwargs.update(kwargs)

        result = self.network.run(latents, None, **run_kwargs)
        if return_w_latents:
            images, dlatents = result
            return self._postprocess(images), dlatents[:, 0, :]
        else:
            images = result
            return self._postprocess(images)

    def run_from_W(self, w_latents, **kwargs):
        """
        Run the generator from the intermediate latent space W

        Inputs:
        :param latents: batch of W-space latent vectors, of the shape [batch_size, latent_dim].
        :param **kwargs: other parameters that will be feeded to the Network.run method of the synthesis network

        Outputs:

        :return images: Batch of generated images in bob format : tensor with shape [batch_size, 3, 1024, 1024] of uint8
                        values in the range [0, 255]
        """
        # Repeat the input latent for each style input
        dlatents = np.tile(w_latents[:, np.newaxis, :], [1, 18, 1])
        run_kwargs = self.run_kwargs
        run_kwargs.update(kwargs)
        return self._postprocess(self.network.components['synthesis'].run(dlatents, **run_kwargs))
    
    def _postprocess(self, images):
        return np.stack([utils.adjust_dynamic_range(img, [0,255], 'uint8') for img in images])


    