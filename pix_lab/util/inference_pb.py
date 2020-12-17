from __future__ import print_function, division

import time
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from scipy import misc
from skimage import measure
from pix_lab.util.util import load_graph
import imutils

class Inference_pb(object):
    """
        Perform inference for an arunet instance

        :param net: the arunet instance to train

        """
    def __init__(self, path_to_pb, img_list, scale=0.33, mode='None'):
        self.graph = load_graph(path_to_pb)
        self.img_list = img_list
        self.scale = scale
        self.mode = mode

    def inference(self, print_result=True, gpu_device="0"):
        # orginal_image = self.img_list
        val_size = len(self.img_list)
        if val_size is None:
            print("No Inference Data available. Skip Inference.")
            return
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.visible_device_list = gpu_device
        with tf.Session(graph=self.graph, config=session_conf) as sess:
            x = self.graph.get_tensor_by_name('inImg:0')
            predictor = self.graph.get_tensor_by_name('output:0')
            print("Start Inference")
            timeSum = 0.0
            imgorg = None
            block_rect = None
            for step in range(0, val_size):
                aTime = time.time()
                aImgPath = self.img_list[step]
                block_img = aImgPath[:-4]+'_block.jpg'
                channel_img = aImgPath[:-4]+'_channel.jpg'
                thresh_img = aImgPath[:-4] + '_thresh.jpg'
                print(
                    "Image: {:} ".format(aImgPath))
                batch_x = self.load_img(aImgPath, self.scale, self.mode)
                print(
                    "Resolution: h {:}, w {:} ".format(batch_x.shape[1],batch_x.shape[2]))
                # Run validation
                aPred = sess.run(predictor,
                                       feed_dict={x: batch_x})
                curTime = (time.time() - aTime)*1000.0
                timeSum += curTime
                print(
                    "Update time: {:.2f} ms".format(curTime))
                if print_result:
                    n_class = aPred.shape[3]
                    channels = batch_x.shape[3]
                    # fig = plt.figure()
                    for aI in range(0, n_class+1):
                        if aI == 0:
                            # a = fig.add_subplot(1, n_class+1, 1)
                            print("")
                            if channels == 1:
                                print("")
                                # import cv2
                                # cv2.imwrite('batch1.jpg', batch_x[0, :, :, 0])
                                # plt.imshow(batch_x[0, :, :, 0], cmap=plt.cm.gray)
                            else:
                                print("")
                                # import cv2
                                # cv2.imwrite('batch2.jpg', batch_x[0, :, :, 0])
                                # plt.imshow(batch_x[0, :, :, :])
                            # a.set_title('input')
                        else:
                            # a = fig.add_subplot(1, n_class+1, aI+1)
                            # plt.imshow(aPred[0,:, :,aI-1], cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
                            misc.imsave(channel_img +'_'+ str(aI) + '.jpg', aPred[0,:, :,aI-1])
                            if aI == 1:
                                imgorg, block_rect = get_lines(aPred,aI,aImgPath,channel_img,thresh_img)
                                cv2.imwrite(block_img,imgorg)
                    print('To go on just CLOSE the current plot.')
                    # plt.show()
            self.output_epoch_stats_val(timeSum/val_size)
            print("Inference Finished!")

            return imgorg, block_rect

    def output_epoch_stats_val(self, time_used):
        print(
            "Inference avg update time: {:.2f} ms".format(time_used))

    def load_img(self, path, scale, mode):
        aImg = misc.imread(path, mode=mode)
        sImg = misc.imresize(aImg, scale, interp='bicubic')
        # sImg = aImg
        fImg = sImg
        if len(sImg.shape) == 2:
            fImg = np.expand_dims(fImg,2)
        fImg = np.expand_dims(fImg,0)

        return fImg

def get_lines(aPred,aI,orginal_image,channel_img,thresh_img):

    imgorg = cv2.imread(orginal_image)
    h_n, w_n = imgorg.shape[:2]
    # batch_x = self.load_img('demo_images/test4.jpg', self.scale, self.mode)
    # open_cv_image = np.array(aPred[0,:, :,aI-1])
    # Convert RGB to BGR
    # open_cv_image = open_cv_image[:, :, ::-1].copy()
    img = cv2.imread(channel_img +'_'+ str(aI) + '.jpg')
    h, w = img.shape[:2]
    img = cv2.resize(img, (w_n, h_n), interpolation=cv2.INTER_AREA)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(imgray, 63, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=4)
    cv2.imwrite(thresh_img, thresh)

    connectivity = 4
    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    block_rect = []
    for i in range(num_labels):
        block_rect.append([(stats[i][0], stats[i][1] - 70), (stats[i][0] + stats[i][2], stats[i][1] + stats[i][3])])
        x = stats[i][0]
        y = stats[i][1] - 70
        w = stats[i][3]
        h = stats[i][2]
        if y and x > 0:

            cv2.rectangle(imgorg, (stats[i][0], stats[i][1]-70), (
                stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]),
                          (0, 255, 0), 4)

            crop = imgorg[y:h + y, x:w + x]
            cv2.imshow("snip", crop)
            cv2.waitKey(0)
    for i in centroids:
        cv2.circle(imgorg, (int(i[0]), int(i[1])), 8, (255, 0, 255), 3)


    return imgorg,block_rect


def get_char():
    return True
