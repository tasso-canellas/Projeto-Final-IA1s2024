# Copyright (c) 2021, Idiap Research Institute. All rights reserved.
#
# This work is made available under a custom license, Non-Commercial Research and Educational Use Only 
# To view a copy of this license, visit
# https://gitlab.idiap.ch/bob/bob.paper.icassp2022_morph_generate/-/blob/master/LICENSE.txt


import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import bob.io.image
import bob.io.base
import modules
import utils as sg_utils
import argparse
# from gridtk.tools import get_array_job_slice

def parse_arguments():
    '''Parses in CLI arguments'''
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('-d', '--dst', default='latents', help='Provide a destination folder path for the results.')
    requiredNamed = parser.add_argument_group('Required arguments')
    requiredNamed.add_argument('-s', '--src', type=check_dir_path, help='Provide the folder path containing the source images.', required=True)
    return parser.parse_args()

def check_dir_path(path):
    '''Checks if the given folder path as an argument exists.'''
    if os.path.isdir(path) or path == 'results':
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path.")

def instantiate_sg2_modules():
    '''Instantiates the SG2 Modules'''
    print('Setting up StyleGAN2 modules')
    # Instantiate the three main modules
    sg_utils.fix_randomness(seed=0)
    cropper = modules.preprocessor.FFHQCropper()
    generator = modules.generator.StyleGAN2()
    projector = modules.projector.Projector(num_steps=1000)
    projector.set_network(generator.network)
    return cropper, projector

def make_latents_path(LATENT_DIR):
    '''Concatenates the `dst_path` and `alpha_val` to create directory to store results.'''
    if not os.path.exists(LATENT_DIR):
        print('Making new directory', LATENT_DIR)
        os.makedirs(LATENT_DIR)

def main():
    '''
    Creates latent vectors of all images given in the src directory.
    '''
    # Parse arguments
    args = parse_arguments()

    # Parameters
    SRC_DIR     = args.src
    LATENT_DIR  = args.dst
    DST_SUFFIX  = '.jpg'
    VEC_SUFFIX  = '.hdf5'

    # Create latents path with verification
    make_latents_path(LATENT_DIR)

    # Don't overwrite existing latents
    existing_files = os.listdir(LATENT_DIR)

    # Iterate through a single image - we use SGE_TASK_ID to parallelize
    list_images = sorted(os.listdir(SRC_DIR))
    # list_images = list_images[get_array_job_slice(len(list_images))]

    # Instantiate the three main modules
    cropper, projector = instantiate_sg2_modules()

    # Loop
    for img in list_images:
        # Ignore data files eg: ._070_08.jpg
        if not img.startswith('.') and img.split(DST_SUFFIX)[0]+VEC_SUFFIX not in existing_files:
            print('Going through file:', img)
            # Load images to convert to latent vector
            ref_images = map(bob.io.image.load, [os.path.join(SRC_DIR, img)])
            # Crop & project images
            crops = list(map(cropper, ref_images))
            results = [projector(crop, verbose=True) for crop in crops]
            # Save generated latent vector as .hdf5
            bob.io.base.save(results[0].w_latent, os.path.join(LATENT_DIR, img.split(DST_SUFFIX)[0]+VEC_SUFFIX))

if __name__ == "__main__":
    main()
