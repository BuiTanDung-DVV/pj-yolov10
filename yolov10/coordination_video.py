import cv2
import numpy as np
import pandas as pd
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
from ultralytics import YOLOv10 as YOLO

# Hàm tính diện tích chồng chéo giữa hai bounding box
def calculate_overlap_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x1 < x2 and y1 < y2:
        overlap_area = (x2 - x1) * (y2 - y1)
    else:
        overlap_area = 0

    return overlap_area

# Hàm xử lý từng frame của video
def process_frame(frame, model, class_names):
    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Tọa độ bounding box
    scores = results[0].boxes.conf.cpu().numpy()  # Độ tin cậy
    class_ids = results[0].boxes.cls.cpu().numpy()  # ID lớp

    data = []
    shapely_boxes = []

    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        xmin, ymin, xmax, ymax = map(int, box)
        class_id = int(class_id)
        object_name = class_names.get(class_id, 'Unknown')

        width = xmax - xmin
        height = ymax - ymin
        area = width * height

        # Vẽ bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Tính diện tích chồng chéo
        overlap_area = 0
        for j, other_box in enumerate(boxes):
            if i != j:
                overlap_area += calculate_overlap_area(box, other_box)

        # Tính tỷ lệ phần trăm diện tích chồng chéo
        percentage = (overlap_area / area * 100) if area > 0 else 0

        # Thêm bounding box vào danh sách shapely box
        shapely_boxes.append(shapely_box(xmin, ymin, xmax, ymax))

        # Làm tròn các giá trị chỉ hiển thị 1 số sau dấu phẩy
        overlap_area = round(overlap_area, 1)
        percentage = round(percentage, 1)
        area = round(area, 1)  # Làm tròn diện tích để không có quá nhiều số sau dấu phẩy

        # Hiển thị thông tin bounding box
        font_scale = 0.6
        font_thickness = 1
        text_color = (0, 255, 0)

        text_top = f"{object_name}: ({xmin}, {ymin}), ({xmax}, {ymax})"
        cv2.putText(frame, text_top, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

        text_bottom = f"Area: {area:.1f} | Overlap: {overlap_area:.1f} | Percent: {percentage:.1f}%"
        cv2.putText(frame, text_bottom, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

        # Lưu thông tin vào danh sách
        data.append({
            'Object Name': object_name,
            'Bounding Box': f"({xmin}, {ymin}), ({xmax}, {ymax})",
            'Area': area,
            'Overlap Area': overlap_area,
            'Percentage': percentage
        })

    # Tính diện tích hợp nhất của tất cả các bounding box
    union_box = unary_union(shapely_boxes)
    total_area = round(union_box.area, 1)  # Làm tròn diện tích hợp nhất
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

        # Lưu dữ liệu vào danh sách tổng hợp
        all_data.extend(data)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")

    # Lưu dữ liệu ra file CSV
    df = pd.DataFrame(all_data)

    # Làm tròn các giá trị trong DataFrame trước khi lưu
    df['Area'] = df['Area'].astype(float).round(1)
    df['Overlap Area'] = df['Overlap Area'].astype(float).round(1)
    df['Percentage'] = df['Percentage'].astype(float).round(1)

    df.to_csv(csv_path, index=False, float_format='%.1f')
    print(f"Data saved to {csv_path}")

# Hàm chính để chạy chương trình
def main(video_path, model, class_names, output_path, csv_path):
    process_video(video_path, model, class_names, output_path, csv_path)

if __name__ == "__main__":
    video_path = 'vdtest/vdtest5.mp4'  # Đường dẫn tới video đầu vào
    output_path = 'processed_video/processed_video5.avi'  # Đường dẫn tới video đầu ra
    csv_path = 'processed_video/processed_data.csv'  # Đường dẫn tới file CSV đầu ra
    model_path = 'runs/detect/train/weights/best.pt'

    class_names = {0: 'Bia_Truc_Bach', 1: 'Coca_Cola', 2: 'Fanta', 3: 'Heiniken'}  # Danh sách các tên lớp

    model = YOLO(model_path)

    main(video_path, model, class_names, output_path, csv_path)
