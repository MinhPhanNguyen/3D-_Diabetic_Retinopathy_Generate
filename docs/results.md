Results

1. Chất lượng tái tạo 3D

Point Cloud: Chất lượng dữ liệu điểm 3D tốt với độ chi tiết cao.

Mesh Reconstruction: Độ chính xác cao trong việc kết nối các điểm để tạo bề mặt liên tục.

Texture Mapping: Ảnh được gán kết cấu rõ nét, nhưng vẫn cần tinh chỉnh màu sắc.

2. Hiệu suất phân loại bệnh lý

Mô hình

Độ chính xác

F1-score

CNN

85%

0.82

Vision Transformer

90%

0.88

GNN (Graph Neural Network)

87%

0.85

Mô hình Vision Transformer đạt độ chính xác cao nhất.

GNN có hiệu suất tốt khi xử lý dữ liệu 3D.

3. Phân tích lỗi

Một số mô hình bị nhầm lẫn giữa các cấp độ bệnh nhẹ và trung bình.

Dữ liệu synthetic từ CycleGAN đôi khi có chất lượng không ổn định.

4. Định hướng cải tiến

Thử nghiệm các phương pháp nâng cao như Hybrid Transformer-CNN.

Cải thiện độ chi tiết của bản đồ độ sâu bằng cách sử dụng MiDaS v3.

Kết hợp thêm dữ liệu chụp cắt lớp OCT để bổ sung thông tin độ sâu chính xác hơn.