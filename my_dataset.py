# -*- coding: utf-8 -*-

'''
CODE 설명
Pytorch 에서 사용하는 dataset 의 형태에 맞게 Data를 만드는 작업
따로 class로 빼놓음
'''
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import cv2
import numpy as np

class NkDataSet(Dataset):

    #초기화 시켜주는 작업

    def __init__(self, csv_path):

        print(csv_path)

        self.to_tensor = transforms.ToTensor()
        self.data_info = pd.read_csv(csv_path,header=None)

        #asarray is convert the input to an array

        self.image_arr = np.asarray(self.data_info.iloc[:,0])
        self.label_arr = np.asarray(self.data_info.iloc[:,1])
        self.data_len = len(self.data_info.index)


    #경로를 통해서 실제 데이터의 접근을 해서 데이터를 돌려주는 함수

    def __getitem__(self, index):

        single_image_name = self.image_arr[index]

        img_as_img = cv2.imread(single_image_name)

        #img size check

        img_as_img = cv2.resize(img_as_img, (100, 100))
        img_as_tensor = self.to_tensor(img_as_img)

        single_image_label = self.label_arr[index]


        return (img_as_tensor, single_image_label)


    #데이터의 전체 길이를 구하는 함수

    def __len__(self):

        return self.data_len
