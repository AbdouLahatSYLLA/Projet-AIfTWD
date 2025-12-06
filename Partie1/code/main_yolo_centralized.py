from ultralytics import YOLO
from torch.optim import Adam


def run_centralized():
    #  load pre-trained model (n = nano,s = small, m = medium)
    # '-cls' stand for classification
    model = YOLO('yolov8s-cls.pt')

    optimizer = Adam(model.parameters(),lr=1e-3)
    # Train
    results = model.train(
        data='dataset_yolo',
        epochs=40,
        imgsz=512,
        batchsz=16,
        optimizer=optimizer,
        project='logs_yolo',
        name='centralized_run',
        # Paramètres anti-overfitting
        dropout=0.3,  # 30% des neurones désactivés aléatoirement
        weight_decay=0.001,  # Régularisation plus forte
        degrees=25.0,  # Rotation +/- 25°
        flipud=0.5,  # Miroir vertical autorisé
        fliplr=0.5,  # Miroir horizontal autorisé
        scale=0.5,  # Zoom +/- 50%
        patience=15
    )

    # 3. Eval
    metrics = model.val()
    print(f"Accuracy: {metrics.top1:.4f}")


if __name__ == '__main__':
    run_centralized()