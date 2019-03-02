# -*- coding: utf-8 -*-

'''
CODE 설명
Pytorch 에서 사용하는 DataSet 의 형태에 맞게 Data를 만드는 작업
따로 class로 빼놓음
PIL을 이용해서 DataSet 을 생성
'''
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import torch

class NkDataSet(Dataset):

    #초기화 시켜주는 작업

    def __init__(self, csv_path):

        self.to_tensor = transforms.ToTensor()
        self.data_info = pd.read_csv(csv_path,header=None)
        #asarray is convert the input to an array
        self.image_arr = np.asarray(self.data_info.iloc[:,0])
        self.label_arr = np.asarray(self.data_info.iloc[:,1])
        self.data_len = len(self.data_info.index)


    #경로를 통해서 실제 데이터의 접근을 해서 데이터를 돌려주는 함수

    def __getitem__(self, index):

        single_image_name = self.image_arr[index]

        img_as_img = Image.open(single_image_name)

        #img size check

        img = img_as_img.resize((100, 100), Image.ANTIALIAS)



        img_as_tensor = self.to_tensor(img_as_img)

        single_image_label = self.label_arr[index]


        return (img_as_tensor, single_image_label)


    #데이터의 전체 길이를 구하는 함수

    def __len__(self):

        return self.data_len


#cav 의 경로를 설정해 줘야 한다.

csv_path = './file/animal_info.csv'

custom_dataset = NkDataSet(csv_path)

#batch size를 2로 하면 에러가 생긴다 . 왜 이런 문제가 생기는 걸까 -> enumerate 는 하나씩 출력을 하기 대문에 문제가 생긴다.

my_dataset_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                batch_size=1,
                                                shuffle=False)


#enumerate 는 list 의 있는 내용을 순서를 매기면서 프린트를 한다.

for i, (images, labels) in enumerate(my_dataset_loader):

    print(images,labels)
