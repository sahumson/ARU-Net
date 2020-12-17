from __future__ import print_function, division
import os
import click

from tools.ARU_Net.pix_lab.util.inference_pb import Inference_pb
from tools.ARU_Net.pix_lab.util.util import read_image_list

# @click.command()
# @click.option('--path_list_imgs', default="./demo_images/imgs.lst")
# @click.option('--path_net_pb', default="./demo_nets/model100_ema.pb")
def run(path_list_imgs, path_net_pb):
    list_inf=[]
    for filepath in os.listdir(path_list_imgs):
        if filepath.endswith('.jpg'):
            file_ocr = os.path.join(path_list_imgs, filepath)
            list_inf.append(file_ocr)
    # list_inf = read_image_list(path_list_imgs)
    inference = Inference_pb(path_net_pb, list_inf, mode='L')
    inference.inference()

if __name__ == '__main__':
    path_list_imgs="./images/"
    path_net_pb="./demo_nets/model100_ema.pb"
    run(path_list_imgs,path_net_pb)