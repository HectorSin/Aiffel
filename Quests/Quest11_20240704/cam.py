# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import copy
import cv2
from PIL import Image

from module.get_data import ds_test
from module.base_code import generate_cam, get_one, visualize_cam_on_image, get_bbox, rect_to_minmax, get_iou

cam_model_path = 'model/cam_model10.h5'

# 모델 불러오기
cam_model = tf.keras.models.load_model(cam_model_path)

# 기본 사진
item = get_one(ds_test)
# CAM 이미지
cam_image = generate_cam(cam_model, item)

# CAM 이미지 시각화
origin_image = item['image'].astype(np.uint8)
cam_image_3channel = np.stack([cam_image*255]*3, axis=-1).astype(np.uint8)
blended_image = visualize_cam_on_image(cam_image_3channel, origin_image)

# bounding box 그리기
rect = get_bbox(cam_image)
b_image = copy.deepcopy(item['image'])
b_image = cv2.drawContours(b_image, [rect], 0, (0, 0, 255), 2)

pred_bbox = rect_to_minmax(rect, item['image'])
iou = get_iou(pred_bbox, item['objects']['bbox'][0])

plt.subplot(2, 2, 1)
plt.imshow(item['image'])
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(cam_image)
plt.title('CAM Image')

plt.subplot(2, 2, 3)
plt.imshow(blended_image)
plt.title('Blended Image')

plt.subplot(2, 2, 4)
plt.imshow(b_image)
plt.title(f'Bounding Box\nIoU: {iou:.2f}')

plt.show()



