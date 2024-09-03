.. image:: morphs_img.png
 :width: 800
 :alt: Morph Faces

====================================================
 Are GAN-based Morphs Threatening Face Recognition?
====================================================

This package contains the source code for generating the different types of morphing attacks used in the experiments of the paper "Are GAN-based Morphs Threatening Face Recognition?"::

    @INPROCEEDINGS{Sarkar_ICASSP_2022,
        author    = {Sarkar, Eklavya and Korshunov, Pavel and Colbois, Laurent and Marcel, Sébastien},
        booktitle = {ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
        title     = {Are GAN-based morphs threatening face recognition?},
        projects  = {Idiap, Biometrics Center},
        year      = {2022},
        pages     = {2959-2963},
        doi       = {10.1109/ICASSP43922.2022.9746477},
        note      = {Accepted for Publication in ICASSP2022},
        pdf       = {http://publications.idiap.ch/attachments/papers/2022/Sarkar_ICASSP_2022.pdf}
    }

Material
---------

- Paper_
- Poster_ 
- Presentation_ 

Installation
------------
This package is part of the signal-processing and machine learning toolbox bob_. 
Install conda_ before continuing.

Download the source code of this paper and unpack it. 
Then, you can create and activate the required conda environment with the following commands::

    $ cd bob.paper.icassp2022_morph_generate
    $ conda install -n base -c conda-forge mamba
    $ mamba env create -f environment.yml -n bob.paper.icassp2022_morph_generate
    $ conda activate bob.paper.icassp2022_morph_generate

This will install all the required software to generate the morphing attacks.


Downloading Models
------------------
The projection process relies on several pre-existing models:

* DLIB Face Landmark detector for cropping and aligning the projected faces exactly as in FFHQ. Example_.
* StyleGAN2_ as the main face synthesis network. We fork the official repository_. The Config-F, trained on FFHQ at resolution 1024 x 1024, is employed.
* A pretrained VGG16_ model, used to compute a perceptual loss between projected and target image.


In order to download those models, one must specify the destination path of choice in the ``~/.bobrc`` file, through the following commands::

    $ bob config set sg2_morph.dlib_lmd_path /path/to/dlib/landmark/detector.dat
    $ bob config set sg2_morph.sg2_path /path/to/stylegan2/pretrained/model.pkl
    $ bob config set sg2_morph.vgg16_path /path/to/vgg16/pretrained/model.pkl

Finally, all the models can be downloaded by running::

    $ python download_models.py

Generating Morphs
------------------
**Note**: StyleGAN2 requires custom GPU-only operations, and at least 12 GB of GPU RAM. Therefore, to run all following examples and perform additional experiments, it is necessary to run this code on a GPU.

The script options can be viewed with::

    $ conda activate bob.paper.icassp2022_morph_generate
    $ python gen_morphs.py -h

The morphs of the following types of morphs can be generated at different alphas:

* OpenCV
* FaceMorpher
* StyleGAN2
* MIPGAN-II

Typical usage::

    $ python gen_morphs.py --opencv --facemorpher --stylegan2 --mipgan2 -s path/to/folder/of/images/ -l path/to/csv/of/pairs.csv -d path/to/destination/folder --latents path/to/latent/vectors --alphas 0.3 0.5 0.7

The ``pairs.csv`` file should simply be a 2 column `.csv` file **without a header** containing only the filenames of the 2 images you want to morph:

* image1.png, image2.png
* image1.png, image3.png

**Note**: Keep in mind that for the ``--stylegan2`` and ``--mipgan2`` arguments, it is necessary to have the latent vectors of all required images generated **beforehand**.

This can be done with the ``gen_latents.py``. Typical usage::

    $ python gen_latents.py -s path/to/folder/of/images/

License
-------

This package uses some components from the `official release of the StyleGAN2 model <https://github.com/NVlabs/stylegan2>`_, which is itself released under the `Nvidia Source Code License-NC <https://gitlab.idiap.ch/bob/bob.paper.icassp2022_morph_generate/-/blob/master/modules/LICENSE.txt>`_, as well as from `OpenCV <https://github.com/spmallick/learnopencv>`_ and `Facemorpher <https://github.com/alyssaq/face_morpher>`_ repositories, both of which are released under a "Non-Commercial Research and Educational Use Only" license.


Contact
-------

For questions or reporting issues to this software package, kindly contact the first author_.

.. _author: eklavya.sarkar@idiap.ch
.. _bob: https://www.idiap.ch/software/bob
.. _conda: https://conda.io
.. _stackoverflow: https://stackoverflow.com/questions/tagged/python-bob
.. _example: http://dlib.net/face_landmark_detection.py.html
.. _StyleGAN2: https://arxiv.org/abs/1912.04958
.. _repository: https://github.com/NVlabs/stylegan2
.. _VGG16: https://arxiv.org/abs/1801.03924
.. _Poster: https://sigport.org/sites/default/files/docs/Sarkar_ICASSP_2022_Morph_Poster.pdf
.. _Paper: https://ieeexplore.ieee.org/document/9746477
.. _Presentation: https://youtu.be/anjDrxQKRhc