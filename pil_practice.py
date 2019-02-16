'''
CODE 설명
PIL를 연습하는 시간
'''

#opencv cv2, PIL
from PIL import Image
import numpy as np

#Image Load
img = Image.open("./data/cat.png")


#Image save
#im.save("cat_2.png")

#Image size check
print(img.size)

#Image resize

#ANTIALIAS 는 높은 성능의 convolution kernel 이다.

img = img.resize((100,100),Image.ANTIALIAS)

print(img.size)

#opencv ,pil 은 양식이 다르다.
#속도는 opencv 가 가장 빠르다. 하지만 사용하는 데이터 형식이 다르다.


#PIL 는 RGB format, Opencv BGR format 이다.
print(type(img))