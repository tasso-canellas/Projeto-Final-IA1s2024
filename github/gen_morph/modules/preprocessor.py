import dlib
import PIL.Image
import bob.io.image
import scipy.ndimage
import numpy as np
from bob.extension import rc

class FFHQCropper(object):
    def __init__(self, dlib_lmd_path=rc['sg2_morph.dlib_lmd_path']):
        """
        Instanciate a face cropper that behaves similarly to the one used to preprocess the FFHQ database.

        :param dlib_lmd_path: Path to the dlib landmark detector model
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dlib_lmd_path)

    def __call__(self, image):
        """
        Run the cropper on the input image.

        :param image: input image in bob format (channels first)
        :return : cropped image in bob format, with shape [3, 1024, 1024]
        """
        # Assuming input image in bob format
        channels_last_img = bob.io.image.to_matplotlib(image)
        landmarks = self.detect_landmarks(channels_last_img)
        cropped = self.crop_and_resize(channels_last_img, lm=landmarks)
        return bob.io.image.to_bob(cropped)

    def detect_landmarks(self, image):
        """
        Run the dlib landmark detector on the input image to build a list of landmarks.

        :param image: input image in matplotlib format (channels last)
        :return: list of 68 landmarks in tuple format : [(x1, y1), (x2, y2), ...]
        """
        detection = self.detector(image, 1)[0]
        shape = self.predictor(image, detection)
        return [(item.x, item.y) for item in shape.parts()]

    def crop_and_resize(self, img, lm, 
                        output_size=1024,
                        transform_size=4096,
                        enable_padding=True):
        """
        Crop, resize and align the image based on the provided landmarks (lm),
        in the same way FFHQ has been preprocessed for training StyleGAN2
        
        This code was entirely borrowed from
        https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py (recreate_aligned_images() function),
        with a few adaptations.
        
        """
        lm = np.array(lm)
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2                

        # Convert to PIL (original code used PIL all the way through)
        img = PIL.Image.fromarray(img)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Done ! Back to numpy format
        img = np.array(img)
        return img