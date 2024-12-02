import torch
import cv2
from gtts import gTTS
import os
import threading

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 음성 재생 중 여부를 나타내는 플래그
is_speaking = False

# 텍스트를 음성으로 변환하고 재생하는 함수
def say_hello():
    global is_speaking
    if is_speaking:  # 이미 음성이 재생 중이면 중단
        return
    is_speaking = True
    try:
        tts = gTTS("안녕하세요", lang="ko")  # 한국어로 변환
        tts.save("hello.mp3")  # 음성 파일 저장
        os.system("start hello.mp3")  # Windows에서 음성 파일 재생 (Linux/Mac은 다른 명령 사용)
    except Exception as e:
        print(f"음성 생성 또는 재생 중 오류 발생: {e}")
    finally:
        is_speaking = False  # 음성 재생이 끝난 후 플래그 해제

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("웹캠이 시작되었습니다. ESC 키를 눌러 종료하세요.")

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

    # 사람 객체가 감지되었는지 확인
    person_detected = False
    for index, row in detections.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        label = row['name']

        # 사각형 그리기
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # 레이블 및 신뢰도 표시
        cv2.putText(frame, f"{label} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 사람이 감지되었을 경우
        if label == "person":
            person_detected = True

    # 사람이 감지되면 음성 재생 (스레드로 실행)
    if person_detected and not is_speaking:
        threading.Thread(target=say_hello).start()

    # 결과를 화면에 표시
    cv2.imshow('YOLOv5 Object Detection', frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
