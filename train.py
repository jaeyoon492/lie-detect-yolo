from ultralytics import YOLO

def train_model():
    # 모델 생성 (YOLOv8n - 가장 작은 모델을 사용)
    model = YOLO("model/yolov8m.pt")  # 또는 yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

    # 모델 학습
    model.train(data="dataset/dataset.yaml", epochs=100, imgsz=640)
    # metrics = model.val()
    #
    # results = model.predict(source="dataset/valid/images/0_background_Blitzcrank_png_jpg.rf.16a02129f99fb5aebce08235d7dd6dc4.jpg", save=True)  # 결과를 이미지로 저장
    model.export(format="onnx")  # ONNX 형식으로 내보내기

if __name__ == '__main__':
    train_model()