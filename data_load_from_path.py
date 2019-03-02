# -*- coding: utf-8 -*-

'''
CODE 설명
현재 경로의 있는 이미지 파일들을 List의 넣기
절대 경로, 상대 경로로 각 각 가져오는 거 구현 해보기
#label 2개
사자, 고양이
여자, 남자
총 10장
'''

import os, os.path
import cv2
import torch


#이미지의 종류들을 분류

image_sort = ['png','jpg','jpeg']
image_list = []

#path 의 어떤 경로를 넣는냐에 따라 상대경로, 절대경로가 결정된다.

path = ''


for filename in os.listdir(path):
    for i in image_sort:

        if filename.lower().endswith(i):

            file_path = path + '/' + filename

            image = cv2.imread(file_path)

            image_list.append(image)


#for check

print(image_list)