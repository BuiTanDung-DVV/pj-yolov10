import cv2
import os
import time

# Mở kết nối đến webcam
cap = cv2.VideoCapture(0)

# Kiểm tra xem webcam có mở thành công không
if not cap.isOpened():
    print("Không thể đọc nguồn camera")
    exit()

# Tạo thư mục đầu ra nếu chưa tồn tại
output_dir = 'images_for_train'
os.makedirs(output_dir, exist_ok=True)

img_counter = 0
capture_interval = 1  # Thời gian giữa các lần chụp ảnh (giây)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không thể đọc khung hình từ webcam")
        break

    # Hiển thị khung hình nếu cần
    cv2.imshow('Webcam', frame)

    # Chụp ảnh sau một khoảng thời gian
    if int(time.time()) % capture_interval == 0:
        img_name = os.path.join(output_dir, "opencv_frame_{}.png".format(img_counter))
        cv2.imwrite(img_name, frame)
        print("{} đã được lưu!".format(img_name))
        img_counter += 1
        time.sleep(1)  # Đợi một giây để tránh việc chụp liên tục cùng một khung hình

    if cv2.waitKey(1) % 256 == 27:  # Phím Escape để thoát
        print("Nhấn Escape, đang đóng...")
        break

# Giải phóng webcam và đóng tất cả các cửa sổ OpenCV
cap.release()
cv2.destroyAllWindows()
