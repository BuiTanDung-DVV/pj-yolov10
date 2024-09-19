from ultralytics import YOLO

# Tạo đối tượng mô hình YOLO với weights đã chọn
model = YOLO('weights/yolov10n.pt')

# Huấn luyện mô hình
model.train(
    data='custom_data.yaml',
    epochs=100,
    batch=16,
    plots=True
)
