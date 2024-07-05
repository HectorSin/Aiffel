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
from module.base_code import generate_grad_cam, get_one, visualize_cam_on_image, get_bbox, rect_to_minmax, get_iou

cam_model_path = 'model/cam_model10.h5'

# 모델 불러오기
cam_model = tf.keras.models.load_model(cam_model_path)

# 기본 사진
item = get_one(ds_test)

# Grad-CAM 이미지
grad_cam5_image = generate_grad_cam(cam_model, 'conv5_block3_out', item)
grad_cam4_image = generate_grad_cam(cam_model, 'conv4_block3_out', item)
grad_cam3_image = generate_grad_cam(cam_model, 'conv3_block3_out', item)
grad_cam2_image = generate_grad_cam(cam_model, 'conv2_block3_out', item)

# Grad-CAM 이미지 시각화
origin_image = item['image'].astype(np.uint8)
grad_cam5_image_3channel = np.stack([grad_cam5_image*255]*3, axis=-1).astype(np.uint8)
grad_cam4_image_3channel = np.stack([grad_cam4_image*255]*3, axis=-1).astype(np.uint8)
grad_cam3_image_3channel = np.stack([grad_cam3_image*255]*3, axis=-1).astype(np.uint8)
grad_cam2_image_3channel = np.stack([grad_cam2_image*255]*3, axis=-1).astype(np.uint8)

blended5_image = visualize_cam_on_image(grad_cam5_image_3channel, origin_image)
blended4_image = visualize_cam_on_image(grad_cam4_image_3channel, origin_image)
blended3_image = visualize_cam_on_image(grad_cam3_image_3channel, origin_image)
blended2_image = visualize_cam_on_image(grad_cam2_image_3channel, origin_image)

# 바운딩 박스 그리기
rect = get_bbox(grad_cam5_image)
b_image = copy.deepcopy(item['image'])
b_image = cv2.drawContours(b_image, [rect], 0, (0, 0, 255), 2)
pred_bbox = rect_to_minmax(rect, item['image'])
iou = get_iou(pred_bbox, item['objects']['bbox'][0])

plt.subplot(2, 4, 1)
plt.imshow(grad_cam2_image)
plt.title('Conv2 Image')

plt.subplot(2, 4, 2)
plt.imshow(grad_cam3_image)
plt.title('Conv3 Image')

plt.subplot(2, 4, 3)
plt.imshow(grad_cam4_image)
plt.title('Conv4 Image')

plt.subplot(2, 4, 4)
plt.imshow(grad_cam5_image)
plt.title('Conv5 Image')

plt.subplot(2, 4, 5)
plt.imshow(blended2_image)
plt.title('Blended Conv2 Image')

plt.subplot(2, 4, 6)
plt.imshow(blended3_image)
plt.title('Blended Conv3 Image')

plt.subplot(2, 4, 7)
plt.imshow(blended4_image)
plt.title('Blended Conv4 Image')

plt.subplot(2, 4, 8)
plt.imshow(blended5_image)
plt.title('Blended Conv5 Image')

plt.show()