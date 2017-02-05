#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Segment images using weights from Fisher Yu (2016). Defaults to
settings for the Pascal VOC dataset.
'''

from __future__ import print_function, division

import argparse
import os

import numpy as np
from PIL import Image
from scipy.ndimage import interpolation

from model import get_model
from utils import interp_map

# Pascal color palette
palette = np.array(
    [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
     [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
     [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]],
    dtype=np.uint8)

input_width, input_height = 900, 900
label_margin = 186


def get_trained_model(args):
    ''' Returns a model with loaded weights. '''

    model = get_model(input_width, input_height)
    weights_data = np.load(args.weights_path).item()

    for layer in model.layers:
        if layer.name in weights_data.keys():
            layer_weights = weights_data[layer.name]
            layer.set_weights((layer_weights['weights'],
                               layer_weights['biases']))

    return model


def forward_pass(args):
    ''' Runs a forward pass to segment the image. '''

    model = get_trained_model(args)

    # Load image and swap RGB -> BGR to match the trained weights
    image_rgb = np.array(Image.open(args.input_path)).astype(np.float32)
    image = image_rgb[:, :, ::-1] - args.mean
    image_size = image.shape

    # Network input shape (batch_size=1)
    net_in = np.zeros((1, input_height, input_width, 3), dtype=np.float32)

    output_height = input_height - 2 * label_margin
    output_width = input_width - 2 * label_margin

    # This simplified prediction code is correct only if the output
    # size is large enough to cover the input without tiling
    assert image_size[0] < output_height
    assert image_size[1] < output_width

    # Center pad the original image by label_margin.
    # This initial pad adds the context required for the prediction
    # according to the preprocessing during training.
    image = np.pad(image,
                   ((label_margin, label_margin),
                    (label_margin, label_margin),
                    (0, 0)), 'reflect')

    # Add the remaining margin to fill the network input width. This
    # time the image is aligned to the upper left corner though.
    margins_h = (0, input_height - image.shape[0])
    margins_w = (0, input_width - image.shape[1])
    image = np.pad(image,
                   (margins_h,
                    margins_w,
                    (0, 0)), 'reflect')

    # Pass the image to the network
    net_in[0] = image
    prob = model.predict(net_in)[0]

    # Upsample
    if args.zoom > 1:
        prob = interp_map(prob, args.zoom, image_size[1], image_size[0])

    # Recover the most likely prediction (actual segment class)
    prediction = np.argmax(prob, axis=2)

    # Apply the color palette to the segmented image
    color_image = np.array(palette)[prediction.ravel()].reshape(
        prediction.shape + (3,))

    print('Saving results to: ', args.output_path)
    Image.fromarray(color_image).save(open(args.output_path, 'w'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', nargs='?', default='images/cat.jpg',
                        help='Required path to input image')
    parser.add_argument('--output_path', default=None,
                        help='Path to segmented image')
    parser.add_argument('--mean', nargs='*', default=[102.93, 111.36, 116.52],
                        help='Mean pixel value (BGR) for the dataset.\n'
                             'Default is the mean pixel of PASCAL dataset.')
    parser.add_argument('--zoom', default=8, type=int,
                        help='Upscaling factor')
    parser.add_argument('--weights_path', default='./dilation_pascal16.npy',
                        help='Weights file')

    args = parser.parse_args()

    if not args.output_path:
        dir_name, file_name = os.path.split(args.input_path)
        args.output_path = os.path.join(
            dir_name,
            '{}_seg.png'.format(
                os.path.splitext(file_name)[0]))

    forward_pass(args)


if __name__ == "__main__":
    main()
