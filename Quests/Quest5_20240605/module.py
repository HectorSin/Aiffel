import os
import urllib
import cv2
import numpy as np
from pixellib.semantic import semantic_segmentation
from matplotlib import pyplot as plt
from PIL import Image
import tempfile

# 모델을 전역으로 로드하여 매번 로드하는 시간을 절약
model_dir = 'models'
model_file = os.path.join(model_dir, 'deeplabv3_xception_tf_dim_ordering_tf_kernels.h5')

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

if not os.path.exists(model_file):
    model_url = 'https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5'
    urllib.request.urlretrieve(model_url, model_file)
    print('모델 다운로드 완료')

model = semantic_segmentation()
model.load_pascalvoc_model(model_file)

def segment_images(img_path, target_class):
    #pascalvoc에서 제공하는 데이터 라벨종류
    LABEL_NAMES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ]
    img_orig = cv2.imread(img_path)

    # 이미지 세그멘테이션
    segvalues, output = model.segmentAsPascalvoc(img_path)

    # 분할된 이미지의 클래스 아이디 출력
    for class_id in segvalues['class_ids']:
        print(LABEL_NAMES[class_id])

    # 컬러맵 생성 및 작업
    colormap = np.zeros((256, 3), dtype = int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    # print(colormap[target_class])

    # BGR 이미지를 RGB로 변환
    seg_color = colormap[target_class][::-1]

    # seg_color = (0, 0, 64)
    # 세그멘테이션된 이미지의 컬러값을 추출
    seg_map = np.all(output==seg_color, axis=-1) 

    # 추출된 컬러값을 이용하여 마스크 생성 작업
    img_show = img_orig.copy()
    img_mask = seg_map.astype(np.uint8) * 255
    color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)
    img_orig_blur = cv2.blur(img_orig, (13,13))


    img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    img_bg_mask = cv2.bitwise_not(img_mask_color)
    img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)

    img_concat = np.where(img_mask_color==255, img_orig, img_bg_blur)
    return img_concat

def segment_cromaky(img_path, target_class):
    #pascalvoc에서 제공하는 데이터 라벨종류
    LABEL_NAMES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ]
    img_orig = cv2.imread(img_path)

    # 이미지 세그멘테이션
    segvalues, output = model.segmentAsPascalvoc(img_path)

    # 분할된 이미지의 클래스 아이디 출력
    for class_id in segvalues['class_ids']:
        print(LABEL_NAMES[class_id])

    # 컬러맵 생성 및 작업
    colormap = np.zeros((256, 3), dtype = int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    # print(colormap[target_class])

    # BGR 이미지를 RGB로 변환
    seg_color = colormap[target_class][::-1]

    # seg_color = (0, 0, 64)
    # 세그멘테이션된 이미지의 컬러값을 추출
    seg_map = np.all(output==seg_color, axis=-1) 

    # 추출된 컬러값을 이용하여 마스크 생성 작업
    img_show = img_orig.copy()
    img_mask = seg_map.astype(np.uint8) * 255
    color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)
    # img_orig_blur = cv2.blur(img_orig, (13,13))
    img_orig_blur = np.full_like(img_orig, (0, 255, 0), dtype=np.uint8)


    img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
    img_bg_mask = cv2.bitwise_not(img_mask_color)
    img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)

    img_concat = np.where(img_mask_color==255, img_orig, img_bg_blur)
    return img_concat


def real_time(frame, target_class):
    LABEL_NAMES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ]

    # numpy 배열을 임시 이미지 파일로 저장
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_image_path = temp_file.name
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_image.save(temp_image_path)

    try:
        # 임시 파일 경로를 사용하여 이미지 세그멘테이션
        segvalues, output = model.segmentAsPascalvoc(temp_image_path)

        for class_id in segvalues['class_ids']:
            print(LABEL_NAMES[class_id])

        colormap = np.zeros((256, 3), dtype=int)
        ind = np.arange(256, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3

        seg_color = colormap[target_class][::-1]
        seg_map = np.all(output == seg_color, axis=-1)

        img_show = frame.copy()
        img_mask = seg_map.astype(np.uint8) * 255
        color_mask = cv2.applyColorMap(img_mask, cv2.COLORMAP_JET)
        img_orig_blur = np.full_like(frame, (0, 255, 0), dtype=np.uint8)

        img_mask_color = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
        img_bg_mask = cv2.bitwise_not(img_mask_color)
        img_bg_blur = cv2.bitwise_and(img_orig_blur, img_bg_mask)

        img_concat = np.where(img_mask_color == 255, frame, img_bg_blur)
    finally:
        os.remove(temp_image_path)

    return img_concat