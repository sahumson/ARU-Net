from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")


import time

import tensorflow as tf
import numpy as np
from scipy import misc

from skimage.transform import radon
from numpy import mean, array, blackman
import numpy
from numpy.fft import rfft
from matplotlib.mlab import rms_flat


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    return graph

class Inference_pb(object):
    """
        Perform inference for an arunet instance

        :param net: the arunet instance to train

        """
    def __init__(self, path_to_pb, img_list, scale=0.33, mode='L'):
        self.graph = load_graph(path_to_pb)
        self.img_list = img_list
        self.scale = scale
        self.mode = mode


    def inference(self, print_result=True, gpu_device="0"):
        rotation_angle = 0
        val_size = len(self.img_list)
        if val_size is None:
            print("No Inference Data available. Skip Inference.")
            return
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.visible_device_list = gpu_device
        with tf.Session(graph=self.graph, config=session_conf) as sess:
            x = self.graph.get_tensor_by_name('inImg:0')
            predictor = self.graph.get_tensor_by_name('output:0')

            timeSum = 0.0
            for step in range(0, val_size):
                aTime = time.time()
                aImgPath = self.img_list[step]
                batch_x = self.load_img(aImgPath, self.scale, self.mode)
                # Run validation
                aPred = sess.run(predictor,
                                       feed_dict={x: batch_x})
                curTime = (time.time() - aTime)*1000.0
                timeSum += curTime

                if print_result:
                    n_class = aPred.shape[3]
                    for aI in range(0, n_class+1):
                        if aI==3:
                            OutIm = aPred[0, :, :, aI - 1]

                            # -*- coding: utf-8 -*-
                            """
                            Automatically detect rotation and line spacing of an image of text using
                            Radon transform
                            If image is rotated by the inverse of the output, the lines will be
                            horizontal (though they may be upside-down depending on the original image)
                            It doesn't work with black borders
                            """
                            try:
                                from parabolic import parabolic
                                def argmax(x):
                                    return parabolic(x, numpy.argmax(x))[0]
                            except ImportError:
                                from numpy import argmax

                            I=OutIm
                            I = I - mean(I)  # Demean; make the brightness extend above and below zero


        return rotation_angle


    def load_img(self, path, scale, mode):
        aImg = misc.imread(path, mode=mode)
        sImg = misc.imresize(aImg, scale, interp='bicubic')
        fImg = sImg
        if len(sImg.shape) == 2:
            fImg = np.expand_dims(fImg,2)
        fImg = np.expand_dims(fImg,0)

        return fImg


def rotation_aru(path_net_pb, list_inf):

    inference = Inference_pb(path_net_pb, [list_inf], mode='L')
    rotation_angle = inference.inference()

    return rotation_angle
