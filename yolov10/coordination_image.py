import cv2
import numpy as np
import pandas as pd
import os
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
from ultralytics import YOLOv10 as YOLO




def process_image(image, model, class_names):
    # Sử dụng YOLO để dự đoán
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    data = []
    shapely_boxes = []

    height = image.shape[0]
    width = image.shape[1]
    area_Of_Image = height * width

    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        xmin, ymin, xmax, ymax = map(int, box)
        class_id = int(class_id)
        object_name = class_names.get(class_id, 'Không xác định')


        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Thêm hộp Shapely để tính diện tích hợp nhất
        shapely_boxes.append(shapely_box(xmin, ymin, xmax, ymax))

        font_scale = 0.6
        font_thickness = 1
        text_color = (0, 255, 0)

        text_top = f"{object_name}: ({xmin}, {ymin}), ({xmax}, {ymax})"
        cv2.putText(image, text_top, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

        # Làm tròn số phần trăm chỉ hiển thị 1 số thập phân



    # Tính diện tích hợp nhất của tất cả các hộp
    union_box = unary_union(shapely_boxes)
    total_area = round(union_box.area)
    percentage = round(total_area/area_Of_Image * 100, 1)

    text_information = f"Total bounding box area : {total_area}  - Percentage : {percentage} %"
    cv2.putText(image, text_information, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
    print(f'Total union area of bounding boxes: {total_area}')

    return image, data



def process_single_image(input_image_path, output_image_path, model, class_names):
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Không thể đọc ảnh tại {input_image_path}")
        return

    processed_image, data = process_image(image, model, class_names)

    # Lưu ảnh đã xử lý
    cv2.imwrite(output_image_path, processed_image)



def main(input_folder, output_folder, model, class_names):
    process_single_image(input_folder, output_folder, model, class_names)


if __name__ == "__main__":
    input_image_path = '../images_for_train/opencv_frame_25.png'  # Thư mục chứa ảnh đầu vào
    output_image_path = '../processed_images/test1.png'  # Thư mục lưu ảnh đầu ra
    csv_folder = 'processed_images/csv_folder'  # Thư mục lưu file CSV đầu ra
    model_path = '../runs/detect/train/weights/best.pt'

    class_names = {0: 'Bia_Truc_Bach', 1: 'Coca_Cola', 2: 'Fanta', 3: 'Heiniken'}

    model = YOLO(model_path)

    main(input_image_path, output_image_path, model, class_names)
