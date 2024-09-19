import cv2
import numpy as np
import pandas as pd
import os
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
from ultralytics import YOLOv10 as YOLO


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


def process_image(image, model, class_names):
    # Sử dụng YOLO để dự đoán
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    data = []
    shapely_boxes = []

    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        xmin, ymin, xmax, ymax = map(int, box)
        class_id = int(class_id)
        object_name = class_names.get(class_id, 'Không xác định')

        width = xmax - xmin
        height = ymax - ymin
        area = width * height

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        overlap_area = 0
        for j in range(len(boxes)):
            if i != j:
                other_box = boxes[j]
                overlap_area += calculate_overlap_area(box, other_box)

        percentage = (overlap_area / area * 100) if area > 0 else 0

        # Giới hạn tỷ lệ phần trăm overlap không vượt quá 100%
        percentage = min(percentage, 100)

        # Thêm hộp Shapely để tính diện tích hợp nhất
        shapely_boxes.append(shapely_box(xmin, ymin, xmax, ymax))

        font_scale = 0.6
        font_thickness = 1
        text_color = (0, 255, 0)

        text_top = f"{object_name}: ({xmin}, {ymin}), ({xmax}, {ymax})"
        cv2.putText(image, text_top, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

        # Làm tròn số phần trăm chỉ hiển thị 1 số thập phân
        # Trong hàm process_image, cập nhật phần hiển thị chỉ số như sau:

        text_bottom = f"Area: {area:.1f} | Overlap: {overlap_area:.1f} | Percent: {percentage:.1f}%"
        cv2.putText(image, text_bottom, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,font_thickness)

        data.append({
            'Object Name': object_name,
            'Bounding Box': f"({xmin}, {ymin}), ({xmax}, {ymax})",
            'Area': area,
            'Overlap Area': overlap_area,
            'Percentage': f"{percentage:.1f}"  # Chuyển thành chuỗi với 1 số thập phân
        })

    # Tính diện tích hợp nhất của tất cả các hộp
    union_box = unary_union(shapely_boxes)
    total_area = union_box.area

    print(f'Total union area of bounding boxes: {total_area}')

    return image, data


def process_images_in_folder(input_folder, output_folder, csv_folder, model, class_names):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)

    all_data = []

    for file_name in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, file_name)
        if os.path.isfile(input_image_path):
            image = cv2.imread(input_image_path)
            if image is not None:
                processed_image, data = process_image(image, model, class_names)

                output_image_path = os.path.join(output_folder, file_name)
                cv2.imwrite(output_image_path, processed_image)

                all_data.extend(data)
            else:
                print(f"Không thể đọc ảnh tại {input_image_path}")

    csv_path = os.path.join(csv_folder, 'processed_data.csv')
    df = pd.DataFrame(all_data)

    # Làm tròn lại cột 'Percentage' trong DataFrame nếu cần
    df['Percentage'] = df['Percentage'].astype(float).round(1)
    df['Area'] = df['Area'].astype(float).round(1)
    df['Overlap Area'] = df['Overlap Area'].astype(float).round(1)

    # Khi lưu CSV, định dạng số thập phân với 1 chữ số
    df.to_csv(csv_path, index=False, float_format='%.1f')
    print(f"Data saved to {csv_path}")


def main(input_folder, output_folder, csv_folder, model, class_names):
    process_images_in_folder(input_folder, output_folder, csv_folder, model, class_names)


if __name__ == "__main__":
    input_folder = 'images_for_train'  # Thư mục chứa ảnh đầu vào
    output_folder = 'processed_images'  # Thư mục lưu ảnh đầu ra
    csv_folder = 'processed_images/csv_folder'  # Thư mục lưu file CSV đầu ra
    model_path = 'runs/detect/train/weights/best.pt'

    class_names = {0: 'Bia_Truc_Bach', 1: 'Coca_Cola', 2: 'Fanta', 3: 'Heiniken'}

    model = YOLO(model_path)

    main(input_folder, output_folder, csv_folder, model, class_names)
