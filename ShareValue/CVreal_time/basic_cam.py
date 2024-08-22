import cv2

# VideoCapture 객체 생성, 0은 기본 웹캠을 의미
cap = cv2.VideoCapture(0)

# 웹캠에서 프레임을 읽어오는 루프
while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    # 프레임이 제대로 읽혔는지 확인
    if not ret:
        break
    
    # 프레임 작업코드 작성 [예시: 프레임 합성]

    # 프레임을 윈도우에 보여주기
    cv2.imshow('Webcam Visualization', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()