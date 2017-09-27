import os
from torch.utils import data
from loader.image_label_loader import imageLabelLoader
from models.deeplab_g2 import deeplabG2
from util.confusion_matrix import ConfusionMatrix
import torch
import numpy as np
import scipy.misc
def color(label):
    bg = label == 0
    bg = bg.reshape(bg.shape[0], bg.shape[1])
    face = label == 1
    face = face.reshape(face.shape[0], face.shape[1])
    hair = label == 2
    hair = hair.reshape(hair.shape[0], hair.shape[1])
    Upcloth = label == 3
    Upcloth = Upcloth.reshape(Upcloth.shape[0], Upcloth.shape[1])
    Larm = label == 4
    Larm = Larm.reshape(Larm.shape[0], Larm.shape[1])
    Rarm = label == 5
    Rarm = Rarm.reshape(Rarm.shape[0], Rarm.shape[1])
    pants = label == 6
    pants = pants.reshape(pants.shape[0], pants.shape[1])
    Lleg = label == 7
    Lleg = Lleg.reshape(Lleg.shape[0], Lleg.shape[1])
    Rleg = label == 8
    Rleg = Rleg.reshape(Rleg.shape[0], Rleg.shape[1])
    dress = label == 9
    dress = dress.reshape(dress.shape[0], dress.shape[1])
    Lshoe = label == 10
    Lshoe = Lshoe.reshape(Lshoe.shape[0], Lshoe.shape[1])
    Rshoe = label == 11
    Rshoe = Rshoe.reshape(Rshoe.shape[0], Rshoe.shape[1])

    # bag = label == 12
    # bag = bag.reshape(bag.shape[0], bag.shape[1])

    # repeat 2nd axis to 3
    label = label.reshape(bg.shape[0], bg.shape[1], 1)
    label = label.repeat(3, 2)
    R = label[:, :, 2]
    G = label[:, :, 1]
    B = label[:, :, 0]
    R[bg] = 230
    G[bg] = 230
    B[bg] = 230

    R[face] = 255
    G[face] = 215
    B[face] = 0

    R[hair] = 80
    G[hair] = 49
    B[hair] = 49

    R[Upcloth] = 51
    G[Upcloth] = 0
    B[Upcloth] = 255

    R[Larm] = 2
    G[Larm] = 251
    B[Larm] = 49

    R[Rarm] = 141
    G[Rarm] = 255
    B[Rarm] = 212

    R[pants] = 160
    G[pants] = 0
    B[pants] = 255

    R[Lleg] = 0
    G[Lleg] = 204
    B[Lleg] = 255

    R[Rleg] = 191
    G[Rleg] = 255
    B[Rleg] = 248

    R[dress] = 255
    G[dress] = 182
    B[dress] = 185

    R[Lshoe] = 180
    G[Lshoe] = 122
    B[Lshoe] = 121

    R[Rshoe] = 202
    G[Rshoe] = 160
    B[Rshoe] = 57

    # R[bag] = 255
    # G[bag] = 1
    # B[bag] = 1
    return label
def update_confusion_matrix(matrix, output, target):
    values, indices = output.max(1)
    output = indices
    target = target.cpu().numpy()
    output = output.cpu().numpy()
    matrix.update(target, output)
    return matrix

def main():
    if len(args['device_ids']) > 0:
        torch.cuda.set_device(args['device_ids'][0])

    test_loader = data.DataLoader(imageLabelLoader(args['data_path'], dataName=args['domainB'], phase='val'),
                                   batch_size=args['batch_size'],
                                   num_workers=args['num_workers'], shuffle=False)
    gym = deeplabG2()
    gym.initialize(args)
    gym.load('/home/ben/mathfinder/PROJECT/AAAI2017/our_Method/v3/deeplab_feature_adaptation/checkpoints/g2_lr_gan=0.00000002_interval_G=5_interval_D=5_net_D=lsganMultOutput_D/best_Ori_on_B_model.pth')
    gym.eval()
    matrix = ConfusionMatrix(args['label_nums'])
    for i, (image, label) in enumerate(test_loader):
        label = label.cuda(async=True)
        target_var = torch.autograd.Variable(label, volatile=True)

        gym.test(image)
        output = gym.output

        matrix = update_confusion_matrix(matrix, output.data, label)
    print(matrix.avg_f1score())
    print(matrix.f1score())


if __name__ == "__main__":
    global args
    args = {
        'test_init':False,
        'label_nums':12,
        'l_rate':1e-8,
        'lr_gan': 0.00000002,
        'beta1': 0.5,
        'interval_G':5,
        'interval_D':5,
        'data_path':'datasets',
        'n_epoch':1000,
        'batch_size':10,
        'num_workers':10,
        'print_freq':100,
        'device_ids':[1],
        'domainA': 'Lip',
        'domainB': 'Indoor',
        'weigths_pool': 'pretrain_models',
        'pretrain_model': 'deeplab.pth',
        'fineSizeH':241,
        'fineSizeW':121,
        'input_nc':3,
        'name': 'train_iou0.4_onehot_g2_lr_gan=0.00000002_interval_G=5_interval_D=5_net_D=lsganMultOutput_D',
        'checkpoints_dir': 'checkpoints',
        'net_D': 'lsganMultOutput_D',
        'use_lsgan': True,
        'resume':None,#'checkpoints/g2_lr_gan=0.0000002_interval_G=5_interval_D=10_net_D=lsganMultOutput_D/best_Ori_on_B_model.pth',#'checkpoints/v3_1/',
        'if_adv_train':True,
        'if_adaptive':True,
    }
    main()