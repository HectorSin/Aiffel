from module import segment_images
from matplotlib import pyplot as plt
import cv2

# 원하는 사진 경로 지정
img_path = 'images/person_image.webp'

# 원하는 대상 지정 [고양이=8, 사람=15, 배경=0]
target_class = 15

# 일반 블러 처리
img_concat = segment_images(img_path, target_class)
plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
plt.show()