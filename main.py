from ultralytics import YOLO
from typing import Any

import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def run_model(image_source: np.ndarray[Any, np.dtype], filename:str):
    # 옵션 설정
    weights_path = 'runs/detect/train8/weights/best.pt'  # 학습된 모델의 경로
    model = YOLO(weights_path)

    # 추론 수행
    confidence_threshold = 0.1
    results = model.predict(source=image_source, imgsz=max(image_source.shape[:2]), conf=confidence_threshold)

    print(f"results {results}")

    result_image = results[0].plot()  # 첫 번째 결과의 시각화된 이미지

    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    plt.imsave(f"detected-images/{filename}", result_image)
    time.sleep(1)


if __name__ == '__main__':
    file_dir = os.listdir("./test-images")
    for image in file_dir:
        image_path = os.path.join("test-images", image)
        image_data = cv2.imread(image_path)
        run_model(image_data, image)
