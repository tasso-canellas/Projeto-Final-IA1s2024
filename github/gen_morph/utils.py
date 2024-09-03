# Copyright (c) 2021, Idiap Research Institute. All rights reserved.
#
# This work is made available under a custom license, Non-Commercial Research and Educational Use Only 
# To view a copy of this license, visit
# https://gitlab.idiap.ch/bob/bob.paper.icassp2022_morph_generate/-/blob/master/LICENSE.txt


import numpy as np
import bob.io.image
import matplotlib.pyplot as plt
from dnnlib import tflib

def fix_randomness(seed=None):
    config = {'rnd.np_random_seed': seed,
            'rnd.tf_random_seed': 'auto'}
    tflib.init_tf(config)

def adjust_dynamic_range(image, target_range, dtype):
    """
    Update the dynamic range of the input image to lie in the required range.
    Example :
    adjust_dynamic_range(image, target_range=[0, 255], dtype='uint8') 
    maps the image to the [0, 255] interval and casts it as a uint8.
    """
    minval = np.min(image)
    maxval = np.max(image)
    return ((target_range[0]*(maxval - image) + target_range[1]*(image-minval))/(maxval - minval)).astype(dtype)

def lerp(p0, p1, n, start=0.0, end=1.0):
    """
    Linear interpolation between two points
    Inputs:
        p0, p1: Rank-1 numpy vectors with same dimension D
        n: int, total number of points to return
        start, end : control the range of the interpolation, i.e. the range of interpolation parameter t.
                    Interpolated points are computed as (1-t)*p0 + t*p1 where t can takes linearly spaced values
                    in the range [start, end] (included).

    Returns:
        p : Numpy vector of shape (n, D) containing all interpolated points
    """
    t = np.linspace(start, end, n)[:, np.newaxis]
    p = (1-t) * p0[np.newaxis, :] + t * p1[np.newaxis, :]
    return p

def facegrid(images, nrows, ncols, figsize=None, labels=None):
    """
    Produces an image grid of size (nrows, ncols) showing each image
    contained in the input `images` list.
    An optional `labels` list can also be provided, in which case the `labels` will be 
    used as title for each subplot in the grid.
    """
    if figsize is None:
        figsize = (2*ncols, 2*nrows)
    if labels is None:
        labels = [None]*len(images)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=figsize, tight_layout=True)
    for i, (face, label) in enumerate(zip(images, labels)):
        currax = ax[i//ncols, i%ncols]
        if face is not None:
            currax.imshow(bob.io.image.to_matplotlib(face))
        currax.axis('off')
        if label is not None:
            currax.set_title(label)
    
    return fig, ax
