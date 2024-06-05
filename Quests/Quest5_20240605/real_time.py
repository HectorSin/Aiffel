from module import real_time
import cv2
import numpy as np

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = real_time(frame, target_class=15, img_path='images/wave.png')  # 예: 'person' 클래스 ID가 15

    cv2.imshow('Video', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 카메라 장치에서 받아온 메모리를 해제하여 카메라 객체 해제
cap.release()
cv2.destroyAllWindows()