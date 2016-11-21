#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Segment images using weights from Fisher Yu (2016). Defaults to
settings for the Pascal VOC dataset.
'''

from __future__ import print_function, division
import os
import cv2
import h5py
import argparse
import numpy as np
from IPython import embed
from keras.preprocessing.image import load_img, img_to_array, array_to_img

from model import get_model
from utils import interp_map


# Pascal color palette
palette = np.array(
    [[0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128],
    [128,128,128],[64,0,0],[192,0,0],[64,128,0],[192,128,0],[64,0,128],[192,0,128],
    [64,128,128],[192,128,128],[0,64,0],[128,64,0],[0,192,0],[128,192,0],[0,64,128]],
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

    image = cv2.imread(args.input_path, 1).astype(np.float32) - args.mean
    image_size = image.shape

    # Shape: (1, 900, 900, 3)
    net_in = np.zeros((1, 900, 900, 3), dtype=np.float32)

    output_height = input_height - 2 * label_margin
    output_width = input_width - 2 * label_margin
    image = cv2.copyMakeBorder(image, label_margin, label_margin,
                               label_margin, label_margin,
                               cv2.BORDER_REFLECT_101)

    # Tile the input to operate on arbitrarily
    # large images.
    num_tiles_h = image_size[0] // output_height + \
                  (1 if image_size[0] % output_height else 0)
    num_tiles_w = image_size[1] // output_width + \
                  (1 if image_size[1] % output_width else 0)

    prediction = []
    for h in range(num_tiles_h):
        col_prediction = []

        for w in range(num_tiles_w):
            offset = [output_height * h,
                      output_width * w]
            tile = image[offset[0]:offset[0] + input_height,
                         offset[1]:offset[1] + input_width, :]
            margin = [0, input_height - tile.shape[0],
                      0, input_width - tile.shape[1]]
            tile = cv2.copyMakeBorder(tile, margin[0], margin[1],
                                      margin[2], margin[3],
                                      cv2.BORDER_REFLECT_101)

            # Pass the tile to the network
            net_in[0] = tile
            prob = model.predict(net_in)[0]
            col_prediction.append(prob)

        col_prediction = np.concatenate(col_prediction, axis=2)
        prediction.append(col_prediction)

    prob = np.concatenate(prediction, axis=1)

    if args.zoom > 1:
        prob = prob.transpose(2, 0, 1)  # to caffe ordering
        prob = interp_map(prob, args.zoom, image_size[1], image_size[0])
        prob = prob.transpose(1, 2, 0)  # to tf ordering

    # Recover the most likely prediction (actual segment class)
    prediction = np.argmax(prob, axis=2)

    # Apply the color palette to the segmented image
    color_image = np.array(palette)[prediction.ravel()].reshape(
        prediction.shape + (3,))
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    print('Writing', args.output_path)
    cv2.imwrite(args.output_path, color_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', nargs='?', default='images/cat.jpg',
                        help='Required path to input image')
    parser.add_argument('--output_path', default=None,
                        help='Path to segmented image')
    parser.add_argument('--mean', nargs='*', default=[102.93, 111.36, 116.52],
                        help='Mean pixel value (BGR) for the dataset.\n'
                             'Default is the mean pixel of PASCAL dataset.')
    parser.add_argument('--zoom', default=8,
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