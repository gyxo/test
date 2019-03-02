# -*- coding: utf-8 -*-

'''
CODE 설명
임의로 데이터 라벨을 매기는 작업이다. 나중에 이 부분은 실제 데이터 라벨로 대치 한다.
'''

import torch

num_classes = 2

#torch.empty 는 n 크기만큼의 공간을 생성한다. random은 num_classes 범위내에서 숫자를 랜덤하게 생성

target = torch.empty(2, dtype=torch.float).random_(num_classes)
