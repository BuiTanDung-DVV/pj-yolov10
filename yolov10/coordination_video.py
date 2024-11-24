import cv2
import numpy as np
import pandas as pd
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
from ultralytics import YOLOv10 as YOLO


# Hàm xử lý từng frame của video
def process_frame(frame, model, class_names):
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Tọa độ bounding box
    scores = results[0].boxes.conf.cpu().numpy()  # Độ tin cậy
    class_ids = results[0].boxes.cls.cpu().numpy()  # ID lớp

    data = []
    shapely_boxes = []

    height = frame.shape[0]
    width = frame.shape[1]
    area_Of_Frame = height * width

    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        xmin, ymin, xmax, ymax = map(int, box)
        class_id = int(class_id)
        object_name = class_names.get(class_id, 'Unknown')

        # Vẽ bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Thêm bounding box vào danh sách shapely box
        shapely_boxes.append(shapely_box(xmin, ymin, xmax, ymax))


        # Hiển thị thông tin bounding box
        font_scale = 0.6
        font_thickness = 1
        text_color = (0, 255, 0)

        text_top = f"{object_name}: ({xmin}, {ymin}), ({xmax}, {ymax})"
        cv2.putText(frame, text_top, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)


    # Tính diện tích hợp nhất của tất cả các bounding box
    union_box = unary_union(shapely_boxes)
    total_area = round(union_box.area, 1)  # Làm tròn diện tích hợp nhất
    percentage = round(total_area/area_Of_Frame * 100, 1)
    text_information = f"Total area of bounding boxes : {total_area} - Percentage : {percentage} %"
    cv2.putText(frame, text_information, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    print(f'Total union area of bounding boxes: {total_area}')

    return frame, data

# Hàm xử lý video đầu vào
def process_video(video_path, model, class_names, output_path, csv_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video at {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    all_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, data = process_frame(frame, model, class_names)
        out.write(processed_frame)
        # Hiển thị frame hiện tại trong cửa sổ có tên 'Video'
        cv2.imshow('Video', frame)
        # Chờ 25ms giữa các frame và thoát khi nhấn phím 'e'
        if cv2.waitKey(25) & 0xFF == ord('e'):
            break

        # Lưu dữ liệu vào danh sách tổng hợp
        all_data.extend(data)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")



# Hàm chính để chạy chương trình
def main(video_path, model, class_names, output_path, csv_path):
    process_video(video_path, model, class_names, output_path, csv_path)

if __name__ == "__main__":
    video_path = '../vdtest/vdtest5.mp4'  # Đường dẫn tới video đầu vào
    output_path = '../processed_video/processed_video5.avi'  # Đường dẫn tới video đầu ra
    csv_path = '../processed_video/processed_data.csv'  # Đường dẫn tới file CSV đầu ra
    model_path = '../runs/detect/train/weights/best.pt'

    class_names = {0: 'Bia_Truc_Bach', 1: 'Coca_Cola', 2: 'Fanta', 3: 'Heiniken'}  # Danh sách các tên lớp

    model = YOLO(model_path)

    main(video_path, model, class_names, output_path, csv_path)
