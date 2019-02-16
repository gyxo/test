# -*- coding: utf-8 -*-

'''
CODE 설명
data_load_from_path 에서 list의 저장한 이미지를 tensor tpye 형변환 시키는 작업
numpy 형태로 먼저 변경 시키고 tensor 로 변환시키는 작업을 거침
'''

import os, os.path
import numpy as np
import cv2
import torch

from torchvision import transforms

#이미의 데이터 형태를 정해주는 작업
image_sort = ['png','jpg','jpeg']
image_list = []

#상대 경로 , 절대 경로를 설정 할 수 있다.
path = '.'

for filename in os.listdir(path):

    for i in image_sort:

        if filename.lower().endswith(i):

            image = cv2.imread(filename)

            #이미지의 크기를 조절해 주는 작업
            re_image = cv2.resize(image,(100,100))

            image_list.append(re_image)

#numpy 형태로 변환
image_arr = np.asarray(image_list)

img_as_tensor = []

# 이미지를 하나씩 tensor로 변경시키는 작업
for i in range(len(image_arr)):

    img_as_tensor.append(transforms.ToTensor()(image_arr[i]))

#tensor로 변경시켰지만 다음과 같이 작업을 하면 문제점이 하나 생긴다.
#왜냐하면 전체 데이터 형이 tensor 이 아니기 때문에!
print(img_as_tensor)
