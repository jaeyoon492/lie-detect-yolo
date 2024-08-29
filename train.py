from ultralytics import YOLO


def train_model():
    # 모델 initialize
    model = YOLO("basemodel/yolov8m.pt")

    # 모델 학습
    model.train(data="dataset/dataset.yaml", epochs=100, imgsz=640)

    # ONNX 형식으로 내보내기
    model.export(format="onnx")


if __name__ == '__main__':
    train_model()
