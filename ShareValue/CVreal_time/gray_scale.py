import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일 필터 적용 [추가]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 그레이스케일 프레임을 윈도우에 보여주기 [추가]
    cv2.imshow('Webcam Visualization - Grayscale', gray_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()