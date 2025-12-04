from ultralytics import YOLO


def run_centralized():
    # 1. load pre-trained model (n = nano)
    # '-cls' stand for classification
    model = YOLO('yolov8n-cls.pt')

    # 2. Train
    results = model.train(
        data='dataset_yolo',
        epochs=25,
        imgsz=224,
        project='logs_yolo',
        name='centralized_run'
    )

    # 3. Eval
    metrics = model.val()
    print(f"Accuracy: {metrics.top1:.4f}")


if __name__ == '__main__':
    run_centralized()