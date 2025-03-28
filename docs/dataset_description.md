📂 raw – Chứa ảnh gốc của đáy mắt (fundus images, back surface).

Đây là dữ liệu chưa qua xử lý, có thể là ảnh từ các nguồn như Kaggle, bệnh viện, hoặc từ OCT scans.

📂 preprocessed – Chứa ảnh đã qua xử lý.

Các bước xử lý có thể bao gồm:

Lọc nhiễu (Noise Reduction)

Cân bằng sáng (Contrast Enhancement)

Chuẩn hóa dữ liệu (Normalization)

Cắt vùng ROI (Region of Interest)

📂 synthetic – Chứa ảnh nhân tạo được tạo ra từ CycleGAN hoặc StyleGAN3.

Mục đích:

Tăng cường dữ liệu (Data Augmentation) bằng cách tạo ra ảnh đáy mắt có đặc điểm giống ảnh thật.

Cải thiện khả năng tổng quát hóa của mô hình.

📂 depth_maps – Chứa bản đồ độ sâu (Depth Maps) từ MiDaS.

Dùng để ước lượng hình dạng 3D của mặt sau nhãn cầu.

Được tạo ra từ ảnh 2D thông qua Depth Estimation Models.

📂 3D_models – Chứa mô hình 3D đã được dựng từ ảnh 2D.

Dữ liệu ở đây có thể bao gồm:

Point Cloud (.ply, .xyz) – Tập hợp các điểm không gian của bề mặt đáy mắt.

Mesh (.obj, .stl, .ply) – Mô hình 3D hoàn chỉnh sau khi chuyển từ point cloud.

📂 annotations – Chứa nhãn (labels) cho dữ liệu.

Ví dụ:

image_001.jpg -> Severe Diabetic Retinopathy

image_002.jpg -> Mild Diabetic Retinopathy

Các nhãn này có thể được sử dụng để huấn luyện mô hình phân loại bệnh lý.

📂 split – Chứa tập dữ liệu sau khi chia thành Train/Validation/Test.

Mỗi thư mục sẽ chứa ảnh đã được chia theo tỷ lệ (VD: 70% Train, 20% Val, 10% Test).