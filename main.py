import torch
import cv2
from gtts import gTTS
import time
import pygame
# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # 웹캠 열기
# cap = cv2.VideoCapture(0)  # 0은 기본 웹캠

# IP 카메라 스트림 URL
ip_camera_url = "rtsp://admin:Admin001@gold33.iptime.org:557/2"

# IP 카메라 연결
cap = cv2.VideoCapture(ip_camera_url)


if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("웹캠이 시작되었습니다. ESC 키를 눌러 종료하세요.")



# 텍스트를 음성으로 변환하고 재생하는 함수
def say_hello():
    try:
        tts = gTTS("안녕하세요", lang="ko")  # 한국어로 변환
        tts.save("hello.mp3")  # 음성 파일 저장
        
        # pygame으로 음성 재생
        pygame.mixer.init()
        pygame.mixer.music.load("hello.mp3")
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():  # 재생 중 대기
            time.sleep(0.1)
    except Exception as e:
        print(f"음성 생성 또는 재생 중 오류 발생: {e}")
    
while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    # OpenCV의 BGR 이미지를 RGB로 변환 (YOLOv5는 RGB 이미지를 사용)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLOv5 모델 추론
    results = model(rgb_frame)

    # 결과를 DataFrame으로 가져오기
    detections = results.pandas().xyxy[0]  # 사물 인식 결과

    # 사물 탐지 결과를 프레임에 그리기    
    for index, row in detections.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        label = row['name']        
        if label == 'person' :
            say_hello()
            


        # 사각형 그리기
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # 레이블 및 신뢰도 표시
        cv2.putText(frame, f"{label} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과를 화면에 표시
    cv2.imshow('YOLOv5 Object Detection', frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
