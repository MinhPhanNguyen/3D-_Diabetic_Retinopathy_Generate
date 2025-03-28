Methodology

1. Tổng quan phương pháp

Dự án này sử dụng kỹ thuật tái tạo ảnh 3D để mô hình hóa bề mặt phía sau của võng mạc từ ảnh fundus. Mục tiêu là tạo ra mô hình 3D giúp phân tích bệnh lý võng mạc do tiểu đường.

2. Các bước xử lý

2.1. Tiền xử lý dữ liệu

Lọc nhiễu: Áp dụng Gaussian Blur hoặc Median Filtering để giảm nhiễu.

Tăng cường độ tương phản: Sử dụng Histogram Equalization hoặc CLAHE.

Phân đoạn ảnh: (Tùy chọn) Tách các mạch máu hoặc vùng tổn thương.

2.2. Ước lượng bản đồ độ sâu

Sử dụng mô hình MiDaS để dự đoán bản đồ độ sâu từ ảnh fundus.

Điều chỉnh và hiệu chỉnh bản đồ độ sâu dựa trên dữ liệu thực tế.

2.3. Tái tạo ảnh 3D

NeRF (Neural Radiance Fields): Mô hình hóa trường ánh sáng để tạo ra ảnh 3D chi tiết.

Structure from Motion (SfM): Xây dựng point cloud từ nhiều ảnh fundus.

Chuyển đổi Point Cloud sang Mesh: Sử dụng thuật toán Poisson Reconstruction.

2.4. Gán kết cấu (Texture Mapping)

Kết hợp ảnh fundus với mô hình 3D để tạo ra ảnh chân thực hơn.

Căn chỉnh màu sắc giữa các vùng ảnh để tránh hiện tượng gián đoạn.

2.5. Huấn luyện mô hình phân loại

Dùng CNN hoặc Vision Transformer để phân loại các cấp độ bệnh võng mạc trên dữ liệu 3D