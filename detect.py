from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 옵션 설정
weights_path = 'runs/detect/train8/weights/best.pt'  # 학습된 모델의 경로
source_path = 'test-files/click3.jpg'  # 검출할 이미지 또는 비디오 파일 경로

model = YOLO(weights_path)

# 이미지 로드
image = cv2.imread(source_path)

# 추론 수행
confidence_threshold = 0.5 # 임계값 설정 (예: 0.5)
results = model.predict(source=image, imgsz=max(image.shape[:2]), conf=confidence_threshold)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result_image = results[0].plot()  # 첫 번째 결과의 시각화된 이미지

plt.imshow(result_image)
plt.axis('off')
plt.show()


