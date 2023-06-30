import os
import random
import numpy as np
import torch
from args_fusion import args
from scipy.misc import imread, imsave, imresize
import matplotlib as mpl
import sys
import math

from os import listdir
from os.path import join

EPSILON = 1e-5

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        elif name.endswith('.tif'):
            images.append(join(directory, file))
        # name1 = name.split('.')
        names.append(name)
    return images, names


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def get_image(path, height=256, width=256, flag=False):
    if flag is True:
        image = imread(path, mode='RGB')
    else:
        image = imread(path, mode='L')

    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='nearest')
    return image


# load images - test phase
def get_test_image(image, height=None, width=None, flag=False):
    # if isinstance(paths, str):
    #     paths = [paths]
    # images = []
    # for path in paths:
        # if flag is True:
        #     image = imread(path, mode='RGB')
        # else:
        #     image = imread(path, mode='L')
    # get saliency part
    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='nearest')

    base_size = 512
    h = image.shape[0]
    w = image.shape[1]
    c = 1
    # if h > base_size or w > base_size:
    #     c = 4
    #     if flag is True:
    #         image = np.transpose(image, (2, 0, 1))
    #     else:
    #         image = np.reshape(image, [1, h, w])
    #     images = get_img_parts(image, h, w)
    # else:
    if flag is True:
        image = np.transpose(image, (2, 0, 1))
    else:
        image = np.reshape(image, [1, image.shape[0], image.shape[1]])
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()

    return images, h, w, c


def get_img_parts(image, h, w):
    images = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    img1 = image[:, 0:h_cen + 3, 0: w_cen + 3]
    img1 = np.reshape(img1, [1, img1.shape[0], img1.shape[1], img1.shape[2]])
    img2 = image[:, 0:h_cen + 3, w_cen - 2: w]
    img2 = np.reshape(img2, [1, img2.shape[0], img2.shape[1], img2.shape[2]])
    img3 = image[:, h_cen - 2:h, 0: w_cen + 3]
    img3 = np.reshape(img3, [1, img3.shape[0], img3.shape[1], img3.shape[2]])
    img4 = image[:, h_cen - 2:h, w_cen - 2: w]
    img4 = np.reshape(img4, [1, img4.shape[0], img4.shape[1], img4.shape[2]])
    images.append(torch.from_numpy(img1).float())
    images.append(torch.from_numpy(img2).float())
    images.append(torch.from_numpy(img3).float())
    images.append(torch.from_numpy(img4).float())
    return images


def recons_fusion_images(img_lists, h, w):
    img_f_list = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    c = img_lists[0][0].shape[1]
    ones_temp = torch.ones(1, c, h, w).cuda()
    for i in range(len(img_lists[0])):
        # img1, img2, img3, img4
        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        img_f = torch.zeros(1, c, h, w).cuda()
        count = torch.zeros(1, c, h, w).cuda()

        img_f[:, :, 0:h_cen + 3, 0: w_cen + 3] += img1
        count[:, :, 0:h_cen + 3, 0: w_cen + 3] += ones_temp[:, :, 0:h_cen + 3, 0: w_cen + 3]
        img_f[:, :, 0:h_cen + 3, w_cen - 2: w] += img2
        count[:, :, 0:h_cen + 3, w_cen - 2: w] += ones_temp[:, :, 0:h_cen + 3, w_cen - 2: w]
        img_f[:, :, h_cen - 2:h, 0: w_cen + 3] += img3
        count[:, :, h_cen - 2:h, 0: w_cen + 3] += ones_temp[:, :, h_cen - 2:h, 0: w_cen + 3]
        img_f[:, :, h_cen - 2:h, w_cen - 2: w] += img4
        count[:, :, h_cen - 2:h, w_cen - 2: w] += ones_temp[:, :, h_cen - 2:h, w_cen - 2: w]
        img_f = img_f / count
        img_f_list.append(img_f)
    return img_f_list


def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
        # img_fusion = img_fusion.cpu().clamp(0, 255).data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion) + EPSILON)
    img_fusion = img_fusion * 255
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    # cv2.imwrite(output_path, img_fusion)
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    # 	img_fusion = imresize(img_fusion, [h, w])
    imsave(output_path, img_fusion)


def get_train_images(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

########## ADDED ##############

import yaml
import cv2

def parse_yaml(yaml_path):
    
    with open(yaml_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    
    return data    

def stereo_correction(IR_path,VIS_path,IR_param,VIS_param,dataset):

    IR_PATH = IR_path
    VIS_PATH = VIS_path

    IR_PARAM_PATH=IR_param
    VIS_PARAM_PATH = VIS_param

    # Parameters
    IR_Param = parse_yaml(IR_PARAM_PATH)
    VIS_Param = parse_yaml(VIS_PARAM_PATH)

    IR_K = np.array(IR_Param["camera_matrix"]["data"],dtype=np.float32).reshape(3,3)
    VIS_K = np.array(VIS_Param["camera_matrix"]["data"],dtype=np.float32).reshape(3,3)

    if dataset=="sthereo":
        IR_distort = np.array(IR_Param["distortion_coefficients"]["data"]).reshape(1,5)
        VIS_distort = np.array(VIS_Param["distortion_coefficients"]["data"]).reshape(1,5)
    else:
        IR_distort = np.array(IR_Param["distortion_coefficients"]["data"]).reshape(1,4)
        VIS_distort = np.array(VIS_Param["distortion_coefficients"]["data"]).reshape(1,4)        
    
    IR_img = cv2.imread(IR_PATH,0)
    VIS_img = cv2.imread(VIS_PATH,0)   

    Thres_H,Thres_W = IR_img.shape
    H,W = VIS_img.shape

    # 빈 이미지 (IR 기준)
    empty_img = np.zeros((Thres_H,Thres_W),dtype='uint8')

    # 들어있는 값
    xvalue = np.linspace(0,W-1,W)
    yvalue = np.linspace(0,H-1,H)

    # 접근에 사용할 index
    x,y = np.meshgrid(xvalue,yvalue)

    if dataset=="sthereo":
    
        IR_2_VIS = [  1.0000,         0,         0,    0.0564,
                0,    1.0000,         0,    -0.0631,
                0,         0,    1.0000,    0.0159,
                0,         0,         0,    1.0000]
        
        IR_2_VIS = np.array(IR_2_VIS).reshape(4,4)

        VIS_2_IR = np.linalg.inv(IR_2_VIS)
        
        IR_img= cv2.undistort(IR_img,IR_K,IR_distort)
        VIS_img = cv2.undistort(VIS_img,VIS_K,VIS_distort)
    
    else:

        IR_2_VIS = [0.9999,0.0041,-0.010,0.0109,
                    -0.0038,0.9997,0.0257,0.0099,
                    0.0106,-0.0256,0.9996,-0.0105,
                    0.0,0.0,0.0,1.0]

        IR_2_VIS = np.array(IR_2_VIS).reshape(4,4)
        VIS_2_IR = np.linalg.inv(IR_2_VIS)
    
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(IR_K,IR_distort,np.eye(3),IR_K,(IR_img.shape[1],IR_img.shape[0]),cv2.CV_32F)
        IR_img = cv2.remap(IR_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(VIS_K,VIS_distort,np.eye(3),VIS_K,(VIS_img.shape[1],VIS_img.shape[0]),cv2.CV_32F)
        VIS_img = cv2.remap(VIS_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) 

    # index로 접근할때는 첫번째가 y(height)
    cord_W = x.flatten().astype(int)
    cord_H = y.flatten().astype(int)
    
    norm_coord = np.array([cord_W,cord_H,np.ones((cord_W.shape))]).reshape(3,-1)
    
    norm_coord = np.linalg.inv(VIS_K) @ norm_coord
    transformed_coord = np.linalg.inv(VIS_2_IR) @ np.vstack((norm_coord,np.ones((cord_W.shape))))
    
    transformed_coord = transformed_coord / transformed_coord[2]

    transformed_coord_pixel = IR_K @ (transformed_coord[0:3])
        
    # 값으로 접근할때는 첫번째가 x(width)
    new_W = transformed_coord_pixel[0,:].astype(int)
    new_H = transformed_coord_pixel[1,:].astype(int)

    mask = (new_H > -1) & (new_H < Thres_H) & (new_W > -1) & (new_W < Thres_W)
        
    empty_img[new_H[mask],new_W[mask]] = VIS_img[cord_H[mask],cord_W[mask]]      

    if dataset=="sthereo":
        IR_transformed = IR_img.astype('uint8')[120:400,:]
        VIS_transformed = empty_img.astype('uint8')[120:400,:]
    else:
        IR_transformed = IR_img.astype('uint8')
        VIS_transformed = empty_img.astype('uint8')
    
    return IR_transformed, VIS_transformed