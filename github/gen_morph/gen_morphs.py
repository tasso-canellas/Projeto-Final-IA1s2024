# Copyright (c) 2021, Idiap Research Institute. All rights reserved.
#
# This work is made available under a custom license, Non-Commercial Research and Educational Use Only 
# To view a copy of this license, visit
# https://gitlab.idiap.ch/bob/bob.paper.icassp2022_morph_generate/-/blob/master/LICENSE.txt


import os
import sys
sys.path.append(os.getcwd())
from src.facemorpher import morpher as fcmorpher
import src.opencv.utils as cv_utils
from bob.extension import rc
from modules import morpher
import utils as sg_utils
from PIL import Image
import pandas as pd
import bob.io.image
import bob.io.base
import numpy as np
import cv2 as cv
import argparse
import modules
import dnnlib

#Implementado em Python 3.7.12
#Para criar morphs com StyleGAN2 e MipGAN2, é necessário primeiro, gerar os vetores latentes com gen_latents.py
def parse_arguments():
    '''Parses in CLI arguments'''
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('-d', '--dst', type=check_dir_path, default='results', help='Provide a destination folder path for the results.')
    parser.add_argument('-a', '--alphas', nargs='+', type=check_float_range, default=[0.5], help="Provide the morphing's alpha values [0, 1] (default: 0.5). Example: --alphas 0.3 0.5 0.7")
    parser.add_argument('--latents', type=check_dir_path, help='Provide the folder path for the latent vectors.')
    requiredNamed = parser.add_argument_group('Required arguments')
    requiredNamed.add_argument('--opencv', action='store_true', help='Morphs using the `opencv` algorithm.')
    requiredNamed.add_argument('--facemorpher', action='store_true', help='Morphs using the `facemorpher` algorithm.')
    requiredNamed.add_argument('--stylegan2', action='store_true', help='Morphs using the `stylegan2` algorithm.')
    requiredNamed.add_argument('--mipgan2', action='store_true', help='Morphs using the `mipgan2` algorithm.')
    requiredNamed.add_argument('-s', '--src', type=check_dir_path, help='Provide the folder path containing the source images.', required=True)
    requiredNamed.add_argument('-l', '--lst', type=check_dir_file, help='Provide the file path of the  `.csv` file containing the names of the pair of images to be morphed.', required=True)
    return parser.parse_args()

def check_float_range(arg, MIN_VAL=0.0, MAX_VAL=1.0):
    '''Type function for argparse - a float within the predefined bounds.'''
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number.")
    if f < MIN_VAL or f > MAX_VAL:
        raise argparse.ArgumentTypeError("Argument must be < " + str(MAX_VAL) + "and > " + str(MIN_VAL))
    return f

def check_dir_path(path):
    '''Checks if the given folder path as an argument exists.'''
    if os.path.isdir(path) or path == 'results':
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path.")

def check_dir_file(path):
    '''Checks if the given file path as an argument exists.'''
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid file.")

def make_dst_path(dst_path, type, alpha_val):
    '''Concatenates the `dst_path` and `alpha_val` to create directory to store results.'''
    full_dst_path = os.path.join(dst_path, type, str(alpha_val))
    if not os.path.exists(full_dst_path):
        print('Making new directory', full_dst_path)
        os.makedirs(full_dst_path)
    return full_dst_path

def make_opencv_morphs(PERMUTATIONS, SRC_DIR, dst_path, detector, predictor, fa, alpha):
    '''
    Loops over all given permutations to generate the opencv morph images.

    Source:
    -------
    Copyright (c) 2016 Satya Mallick <spmallick@learnopencv.com>
    All rights reserved. No warranty, explicit or implicit, provided.
    https://learnopencv.com
    '''
    print('Generating OpenCV morphs with alpha', alpha)
    # Loop
    for f1, f2 in PERMUTATIONS:
        print('Morphing files:', f1, f2)
        # Read images
        img1 = np.array(Image.open(os.path.join(SRC_DIR, f1)))
        img2 = np.array(Image.open(os.path.join(SRC_DIR, f2)))
        # Convert from BGR to RGB
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
        # Get grayscale images
        gray1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
        # Get rectangles
        rects1 = detector(img1, 1)
        rects2 = detector(img2, 1)
        # Align images
        img1 = fa.align(img1, gray1, rects1[0])
        img2 = fa.align(img2, gray2, rects2[0])
        # We need the landmarks again as we have changed the size
        rects1 = detector(img1, 1)
        rects2 = detector(img2, 1)
        # Extract landmarks
        points1 = predictor(img1, rects1[0])
        points2 = predictor(img2, rects2[0])
        points1 = cv_utils.face_utils.shape_to_np(points1)
        points2 = cv_utils.face_utils.shape_to_np(points2)
        points = []
        # Compute weighted average point coordinates
        for i in range(0, len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            points.append((x, y))
        # Allocate space for final output
        imgMorph = np.zeros(img1.shape, dtype=img1.dtype)
        # Rectangle to be used with Subdiv2D
        size = img1.shape
        rect = (0, 0, size[1], size[0])
        # Create an instance of Subdiv2D
        subdiv = cv.Subdiv2D(rect)
        d_col = (255, 255, 255)
        # Calculate and draw delaunay triangles
        delaunayTri = cv_utils.calculateDelaunayTriangles(
            rect, subdiv, points, img1, 'Delaunay Triangulation', d_col, draw=False)
        # Morph by reading calculated triangles
        for line in delaunayTri:
            x, y, z = line
            x = int(x)
            y = int(y)
            z = int(z)
            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [points[x],  points[y],  points[z]]
            # Morph one triangle at a time.
            cv_utils.morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)
        # Remove the black
        for i in range(len(imgMorph)):
            for j in range(len(imgMorph[i])):
                if not np.any(imgMorph[i][j]):
                    imgMorph[i][j] = (1.0 - alpha) * \
                        img1[i][j] + alpha * img2[i][j]
        # Save morphed image
        newname = os.path.join(dst_path, f1.replace('.jpg', '') + '_' + f2.replace('.jpg', '')+'_opencv.jpg')
        print(newname, imgMorph.shape)
        cv.imwrite(newname, imgMorph)

def make_facemorpher_morphs(PERMUTATIONS, SRC_DIR, WIDTH, HEIGHT, dst_path, alpha):
    '''
    Loops over all given permutations to generate the facemorph morph images.

    Source:
    -------
    This code is based on the original one by Alyssa Quek, cloned from the face_morpher repository.
    To view the source repository of this code, visit:
    https://github.com/alyssaq/face_morpher
    '''
    print('Generating FaceMorpher morphs with alpha', alpha)
    # Loop
    for f1, f2 in PERMUTATIONS:
        print('Morphing files:', f1, f2)
        fcmorpher.morpher(imgpaths=[os.path.join(SRC_DIR, f1), os.path.join(SRC_DIR, f2)],
                          width=WIDTH,
                          height=HEIGHT,
                          num_frames=12,
                          out_frames=dst_path,
                          background='average',
                          alpha=alpha)

def fix_randomness():
    '''Fixes np and tf seed.'''
    sg_utils.fix_randomness(seed=0)

def instantiate_generator():
    '''Instantiates SG2 Generator'''
    print('Setting up StyleGAN2 modules')
    return modules.generator.StyleGAN2()

def instantiate_cropper():
    '''Instantiates SG2 Cropper'''
    return modules.preprocessor.FFHQCropper()

def check_for_latents(PERMUTATIONS, LATENTS_DIR, vec_suffix='.hdf5'):
    '''Checks if all required latent vectors are present in the `LATENTS_DIR` directory'''
    print('Checking for all existing latents.')
    latents_lst = os.listdir(LATENTS_DIR)
    missing = False
    for f1, f2 in PERMUTATIONS:
        if f1[:-4]+vec_suffix not in latents_lst:
            print('Missing latent:', f1)
            missing = True
        if f2[:-4]+vec_suffix not in latents_lst:
            print('Missing latent:', f2)
            missing = True
    if missing:
        sys.exit("Please generate all the necessary latent vectors of the images to morph before running the StyleGAN2 morphing algorithm.")

def make_stylegan2_morphs(PERMUTATIONS, DST_SUFFIX, LATENTS_DIR, arg_dst_path, generator, ALPHA_LIST, vec_suffix='.hdf5'):
    '''Loops over all given permutations to generate the stylegan2 morph images.'''
    print('Generating StyleGAN2 morphs')
    # Loop
    for f1, f2 in PERMUTATIONS:
        # Load projected images from existing .hdf5 files
        latents_path = [os.path.join(LATENTS_DIR, f1[:-4] + vec_suffix),
                        os.path.join(LATENTS_DIR, f2[:-4] + vec_suffix)]
        latents = list(map(bob.io.image.load, latents_path))
        # Interpolate
        morph_lats = []
        for alpha in ALPHA_LIST:
            morph_lats.append(latents[0] * (1-alpha) + latents[1] * alpha)
        # Stack
        w_latents = np.stack(morph_lats)
        # Generated associated interpolated images
        lerp_images = generator.run_from_W(w_latents)
        # Save the morphed image
        for i, img in enumerate(lerp_images):
            dst_path = make_dst_path(arg_dst_path, 'stylegan2', ALPHA_LIST[i])
            newname = os.path.join(dst_path, f1.replace('.jpg', '') + '_' + f2.replace('.jpg', '')+'_'+str(ALPHA_LIST[i])+'_stylegan2.jpg')
            bob.io.base.save(img, newname)

def make_mipgan2_morphs(PERMUTATIONS, DST_SUFFIX, SRC_DIR, LATENTS_DIR, arg_dst_path, generator, cropper, alpha):
    '''
    Loops over all given permutations to generate the mipgan2 morph images.

    Source:
    -------
    The implementation is a modified version of the one described in:
    Zhang, H., Venkatesh, S., Ramachandra, R., Raja, K., Damer, N. and Busch, C., 2021. 
    Mipgan—generating strong and high quality morphing attacks using identity prior driven gan. 
    IEEE Transactions on Biometrics, Behavior, and Identity Science, 3(3), pp.365-383.
    URL: https://arxiv.org/abs/2009.01729
    '''
    # Get Morpher class, and set network (different init to Projector class)
    morph = morpher.Morpher(alpha=alpha)
    morph.set_network(generator.network)
    # Morph pair-by-pair
    for f1, f2 in PERMUTATIONS:
        # Create morph name
        f1 = f1[:-4]
        f2 = f2[:-4]
        m_name = '_'.join((f1, f2)) + '_' + str(alpha) +'_mipgan2'+ DST_SUFFIX
        # Load, crop, and process images
        x1 = cropper(bob.io.image.load(os.path.join(SRC_DIR, f1+DST_SUFFIX)))
        x2 = cropper(bob.io.image.load(os.path.join(SRC_DIR, f2+DST_SUFFIX)))
        pair_images = np.array([x1, x2])
        pair_images = modules.misc.adjust_dynamic_range(pair_images, [0, 255], [-1, 1])
        # Morph
        morph.start(pair_images, LATENTS_DIR, (f1, f2))
        while morph.get_cur_step() < morph.num_steps:
            morph.step()
        dst_path = make_dst_path(arg_dst_path, 'mipgan2', alpha)
        newname = os.path.join(dst_path, m_name)
        modules.misc.save_image_grid(morph.get_images_interp(), newname, drange=[-1,1])
        #print('\r%-30s\r' % '', end='', flush=True)

def main():
    '''
    Makes OpenCV morphs between selected images given in the `.csv` file.
    '''
    # Parse arguments
    args = parse_arguments()

    # Define variables
    PERMUTATIONS  = pd.read_csv(args.lst, header=None).values
    DLIB_LMD_PATH = rc['sg2_morph.dlib_lmd_path']
    SRC_DIR       = args.src
    ALPHA_LIST    = [0.5]
    if args.latents:
        LATENTS_DIR = args.latents

    SRC_SUFFIX    = '.jpg'
    DST_SUFFIX    = '.jpg'
    WIDTH         = 360
    HEIGHT        = 480

    # Fix seed
    if args.stylegan2 or args.mipgan2:
        fix_randomness()

    # Instantiate dlib detector and predictors
    print('Instantiating modules.')
    detector  = cv_utils.dlib.get_frontal_face_detector()
    predictor = cv_utils.dlib.shape_predictor(DLIB_LMD_PATH)
    fa        = cv_utils.FaceAligner(predictor, desiredFaceWidth=WIDTH, desiredFaceHeight=HEIGHT)

    # OpenCV Morphs
    if args.opencv:
        for alpha in ALPHA_LIST:
            dst_path = make_dst_path(args.dst, 'opencv', alpha)
            make_opencv_morphs(pd.read_csv(args.lst.replace('.csv', '_'+str(alpha).replace('.','')+'.csv'), header=None).values, SRC_DIR, dst_path, detector, predictor, fa, alpha)

    # FaceMorpher Morphs
    if args.facemorpher:
        for alpha in ALPHA_LIST:
            dst_path = make_dst_path(args.dst, 'facemorpher', alpha)
            make_facemorpher_morphs(pd.read_csv(args.lst.replace('.csv', '_'+str(alpha).replace('.','')+'.csv'), header = None).values, SRC_DIR, WIDTH, HEIGHT, dst_path, alpha)

    # StyleGAN2 Morphs - we can one shot for all alphas
    if args.stylegan2:
        check_for_latents(PERMUTATIONS, LATENTS_DIR)
        generator = instantiate_generator()
        make_stylegan2_morphs(PERMUTATIONS, DST_SUFFIX, LATENTS_DIR, args.dst, generator, ALPHA_LIST)

    # MIPGAN-II Morphs
    if args.mipgan2:
        check_for_latents(PERMUTATIONS, LATENTS_DIR)
        cropper = instantiate_cropper()
        if not args.stylegan2:
            generator = instantiate_generator()
        for alpha in ALPHA_LIST:
            make_mipgan2_morphs(PERMUTATIONS, DST_SUFFIX, SRC_DIR, LATENTS_DIR, args.dst, generator, cropper, alpha)
    
    # Finish
    print('Job completed !')

if __name__ == "__main__":
    main()
