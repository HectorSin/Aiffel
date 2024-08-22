# 웹캠활용 실시간 모델 구현

# 어떻게 웹캠으로 시각화를 하는가

OpenCV는 컴퓨터 비전 작업을 위한 오픈 소스 라이브러리입니다. VideoCapture는 OpenCV의 클래스 중 하나로, 카메라나 비디오 파일에서 프레임을 캡처할 수 있습니다.

# 시연

디텍션과 함께하는 사진 합성, Segmentation과 함께하는 배경 합성

# OpenCV VideoCapture 들어가보기

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/951def4c-f0e4-4502-a709-883ee24ea7c8/ce9adbe7-ff8d-4dd8-989b-b35afc27022a/Untitled.png)

OpenCV에서 카메라랑 동영상으로부터 Frame(프레임)을 받아오는 작업을 cv2.VideoCapture 클래스 하나로 처리합니다.

# 약간의 코드 설명

## 1. 카메라 열기

cv2.VideoCapture() 명령어로 카메라를 열 수 있습니다.

들어가는 요소

```python
cv2.VideoCapture(index, apiPreference=None) -> retval
```

**index** : camera_id + domain_offset_id 시스템 기본 카메라를 기본 방법으로 열려면 index에 0을 전달합니다. 장치관리자에 등록되어 있는 카메라 순서대로 인덱스가 설정되어 있습니다.

**apiPreference** : 선호하는 카메라 처리 방법을 지정합니다.

**retval** : cv2.VideoCapture 객체를 반환합니다.

## 1. 비디오 열기

OpenCV를 이용해서 동영상 여는 방법은 카메라 여는 방법과 동일합니다.

차이점은 cv2.VideoCapture() 안에 인덱스 대신에 파일명을 넣어주면 됩니다.

```python
cv2.VideoCapture(filename, apiPreference=None) -> retval
```

**filename** : 비디오 파일 이름, 정지 영상 시퀀스, 비디오 스트림 URL 등, ex) 'video.avi', 'img_%02d.jpg', 'protocol://host:port/script?params|auth'

**apiPreference** : 선호하는 카메라 처리 방법을 지정합니다.

**retval** : cv2.VideoCapture 객체를 반환합니다.

예시 코드

```python
cap = cv2.VideoCapture('video1.mp4')
```

## 2. 비디오 캡쳐 준비여부 확인

## 3. 프레임 받아오기 - **cv2.VideoCapture.read()**

read() 명령어를 통해 카메라, 동영상에서 프레임 받아오기 가능

```python
cv2.VideoCapture.read(image=None) -> retval, image
```

**retval** : 성공하면 True, 실패하면 False.

**image** : 현재 프레임 (numpy.ndarray)

```python
# 비디오 매 프레임 처리
while True: # 무한 루프
    ret, frame = cap.read() # 두 개의 값을 반환하므로 두 변수 지정

    if not ret: # 새로운 프레임을 못받아 왔을 때 braek
        break
        
    # 정지화면에서 윤곽선을 추출
    edge = cv2.Canny(frame, 50, 150)
    
    inversed = ~frame  # 반전

    cv2.imshow('frame', frame)
    cv2.imshow('inversed', inversed)
    cv2.imshow('edge', edge)

    # 10ms 기다리고 다음 프레임으로 전환, Esc누르면 while 강제 종료
    if cv2.waitKey(10) == 27:
        break

cap.release() # 사용한 자원 해제
cv2.destroyAllWindows()
```

## 3. 특수케이스 [모델이 ndarray을 받아들이지 못하는 경우]

```python
# 임시 파일을 생성합니다. 이 파일은 자동으로 닫히고, delete=False 옵션은 파일을 닫은 후에도 삭제되지 않도록 설정합니다.
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_image_path = temp_file.name
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_image.save(temp_image_path)
```