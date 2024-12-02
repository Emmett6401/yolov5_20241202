import cv2

# 웹캠 캡처 객체 생성
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("웹캠이 시작되었습니다. ESC 키를 눌러 종료하세요.")

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 프레임 표시
    cv2.imshow('Webcam', frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
