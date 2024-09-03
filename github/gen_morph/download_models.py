# Copyright (c) 2021, Idiap Research Institute. All rights reserved.
#
# This work is made available under a custom license, Non-Commercial Research and Educational Use Only 
# To view a copy of this license, visit
# https://gitlab.idiap.ch/bob/bob.paper.icassp2022_morph_generate/-/blob/master/LICENSE.txt


import os
import urllib.request
import bz2

from bob.extension import rc

DLIB_LMD_PATH = rc['sg2_morph.dlib_lmd_path']
SG2_PATH = rc['sg2_morph.sg2_path']
VGG16_PATH = rc['sg2_morph.vgg16_path']

parent_dir = '/home/tasso/morph_generate'
directory = 'c'
path = os.path.join(parent_dir, directory)

def makedirs(path):
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)

def download_dlib_lmd():
    dlib_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    if not os.path.exists(DLIB_LMD_PATH):
        makedirs(DLIB_LMD_PATH)

        print('Downloading dlib face landmarks detector...')
        tmp_file, _ = urllib.request.urlretrieve(dlib_url)
        with bz2.BZ2File(tmp_file, 'rb') as src, open(DLIB_LMD_PATH, 'wb') as dst:
            dst.write(src.read())
        print("Success !")
    else:
        print('dlib landmark detector already downloaded in {}'.format(DLIB_LMD_PATH))

def download_stylegan2():
    stylegan2_url = 'http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl'
    if not os.path.exists(SG2_PATH):
        makedirs(SG2_PATH)
        print('Downloading pretrained StyleGAN2 (FFHQ-config-f)...')
        dst_file, _ = urllib.request.urlretrieve(stylegan2_url, SG2_PATH)

        print("Success !")
    else:
        print('StyleGAN2 model already downloaded in {}'.format(SG2_PATH))

def download_vgg16():
    vgg16_url = "http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/vgg16_zhang_perceptual.pkl"
    if not os.path.exists(VGG16_PATH):
        makedirs(VGG16_PATH)
        print("Downloading pretrained VGG16...")
        dst_file, _ = urllib.request.urlretrieve(vgg16_url, VGG16_PATH)
        print("Success !")
    else:
        print("VGG16 model already downloaded in {}".format(VGG16_PATH))

def download_models():
    download_dlib_lmd()
    download_stylegan2()
    download_vgg16()

if __name__ == "__main__":
    download_models()
