import cv2
import numpy as np
import tensorflow as tf
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union


def preprocess_image(image, input_shape):
    # Resize và chuẩn hóa ảnh
    resized_image = cv2.resize(image, (input_shape[1], input_shape[2]))
    normalized_image = resized_image / 255.0  # Chuẩn hóa pixel về [0, 1]
    return np.expand_dims(normalized_image, axis=0).astype(np.float32)


def predict_with_tflite(image, interpreter, input_details, output_details):
    # Lấy kích thước đầu vào
    input_shape = input_details[0]['shape']
    # Tiền xử lý ảnh
    processed_image = preprocess_image(image, input_shape)
    # Gán ảnh vào đầu vào của mô hình
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    # Thực hiện dự đoán
    interpreter.invoke()
    # Lấy kết quả đầu ra
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def postprocess_output(output_data, original_shape):
    boxes, confidences, class_ids = [], [], []
    for detection in output_data[0]:
        confidence = detection[4]  # Xác suất đối tượng
        if confidence > 0.25:  # Ngưỡng confidence
            xmin, ymin, xmax, ymax = (detection[0:4] * np.array([
                original_shape[1], original_shape[0],
                original_shape[1], original_shape[0]
            ])).astype(int)
            class_id = int(detection[5])  # Lớp đối tượng
            boxes.append([xmin, ymin, xmax, ymax])
            confidences.append(confidence)
            class_ids.append(class_id)
    return boxes, confidences, class_ids


def process_image(image, interpreter, input_details, output_details, class_names):
    # Gọi dự đoán từ mô hình TFLite
    output_data = predict_with_tflite(image, interpreter, input_details, output_details)
    # Hậu xử lý kết quả
    boxes, confidences, class_ids = postprocess_output(output_data, image.shape)

    shapely_boxes = []
    height, width, _ = image.shape
    area_of_image = height * width

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        xmin, ymin, xmax, ymax = box
        object_name = class_names.get(class_id, 'Không xác định')
        # Vẽ bounding box lên ảnh
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Vẽ tên lớp và xác suất
        cv2.putText(image, f"{object_name}: {confidence:.2f}", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # Thêm hộp Shapely để tính diện tích hợp nhất
        shapely_boxes.append(shapely_box(xmin, ymin, xmax, ymax))

    # Tính diện tích hợp nhất của tất cả các hộp
    union_box = unary_union(shapely_boxes)
    total_area = round(union_box.area)
    percentage = round(total_area / area_of_image * 100, 1)

    # Hiển thị thông tin diện tích trên ảnh
    text_information = f"Total bounding box area: {total_area} - Percentage: {percentage}%"
    cv2.putText(image, text_information, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    print(f'Total union area of bounding boxes: {total_area}')

    return image


def main(input_image_path, output_image_path, interpreter, input_details, output_details, class_names):
    # Đọc ảnh đầu vào
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Không thể đọc ảnh tại {input_image_path}")
        return

    # Xử lý ảnh
    processed_image = process_image(image, interpreter, input_details, output_details, class_names)

    # Lưu ảnh đã xử lý
    cv2.imwrite(output_image_path, processed_image)
    print(f"Ảnh đã được lưu tại {output_image_path}")


if __name__ == "__main__":
    # Đường dẫn tới file ảnh
    input_image_path = 'images_for_train/opencv_frame_25.png'
    output_image_path = 'processed_images/tests1.png'
    # Đường dẫn mô hình TFLite
    tflite_model_path = 'runs/detect/train3/weights/best_saved_model/best_float32.tflite'

    # Danh sách lớp
    class_names = {0: 'Bia_Truc_Bach', 1: 'Coca_Cola', 2: 'Fanta', 3: 'Heiniken'}

    # Khởi tạo TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Chạy chương trình
    main(input_image_path, output_image_path, interpreter, input_details, output_details, class_names)
