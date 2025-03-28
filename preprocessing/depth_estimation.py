import cv2
import numpy as np
from pathlib import Path
from utils.config import config
from utils.logger import logger
from skimage import measure
from scipy.ndimage import gaussian_filter
import os
import pandas as pd
import matplotlib.pyplot as plt

"""
    Phân Tích Hình Ảnh Bản Đồ Sâu và Thống Kê
    Trực Quan Hóa Bản Đồ Sâu
    Hình ảnh hiển thị một bản đồ sâu với nhãn "1006_depth_map.jpg", trong đó:
    Có một dải màu chuyển từ tối (0) đến sáng (255) biểu diễn giá trị độ sâu.
    Vùng tối hơn chỉ ra độ sâu lớn hơn (xa hơn so với người quan sát).
    Vùng sáng hơn chỉ ra độ sâu nhỏ hơn (gần hơn so với người quan sát).
    Thang Đo Số Liệu
    Hình ảnh bao gồm một số thang đo số liệu, có thể được giải thích như sau:
    Thang độ sâu chính (bên trái):

    250 (gần nhất)
    200
    150
    100
    50
    0 (xa nhất)

    Thang đo phụ (bên phải và bên dưới):
    Có vẻ là tọa độ điểm ảnh theo trục x và y
    Phạm vi: 0-500 theo chiều ngang, 0-250 theo chiều dọc
    Phân Tích Thống Kê
    Biểu đồ tần suất và các thống kê quan trọng tiết lộ đặc điểm của ảnh:
    Các chỉ số chính:

    Min: 1 (pixel tối nhất không phải nền đen)
    Max: 253 (pixel sáng nhất)
    Mean: 191.95 (giá trị trung bình nghiêng về độ sâu gần hơn)
    Median: 217.00 (giá trị giữa, cao hơn trung bình, cho thấy có một số pixel rất tối)
    Độ lệch chuẩn: 60.81 (độ biến thiên trung bình)
    Phạm vi: 252 (độ chênh lệch rộng)
    Pixel khác 0 (%): 59.79 (khoảng 40% là nền đen)
    Entropy: 7.24 (độ phức tạp cao của mẫu độ sâu)

    Giải thích biểu đồ tần suất:
    Phân bố hai đỉnh (bimodal) với điểm tập trung ở cả vùng sáng và tối.
    Số lượng pixel đáng kể trải dài trên hầu hết các mức độ sâu.
    Rất ít pixel hoàn toàn đen (0) hoặc trắng (255).

    Ý Nghĩa Lâm Sàng/Kỹ Thuật
    Phân biệt độ sâu tốt: Khoảng giá trị rộng (1-253) cho thấy sự khác biệt rõ ràng giữa các mức độ sâu.
    Tách biệt nền: 40% pixel nền đen có thể đại diện cho vùng ngoài võng mạc.
    Vùng tập trung chính: Giá trị trung vị cao hơn trung bình cho thấy vùng võng mạc chính gần hơn (sáng hơn).
    Chất lượng hình ảnh: Entropy cao (7.24) cho thấy độ sâu phong phú nhưng không bị nhiễu.

    Kết Luận
    Đây có vẻ là một bản đồ độ sâu thành công của ảnh mắt, trong đó:
    Cấu trúc đĩa thị/đĩa quang được phân biệt rõ ràng.
    Các độ sâu khác nhau của võng mạc được ghi nhận tốt.
    Nền được loại bỏ một cách hợp lý.
    Các chuyển đổi độ sâu mượt mà và có tính hợp lý về mặt lâm sàng.
"""

class RetinalDepthEstimator:
    def __init__(self):
        self.processed_pairs = set()
        self.analysis_results = []  # Lưu kết quả phân tích
        self.display_samples = 5    # Số lượng mẫu hiển thị
        
    def process_pair(self, left_path, right_path, output_dir):
        """Xử lý một cặp ảnh left-right và lưu depth map"""
        # Lấy ID từ tên file (ví dụ: "0" từ "0_left.jpg")
        pair_id = os.path.basename(left_path).split('_')[0]
        
        if pair_id in self.processed_pairs:
            logger.info(f"Đã xử lý cặp {pair_id}, bỏ qua")
            return None
            
        # Đọc ảnh và chuyển đổi
        left_img = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        
        if left_img is None or right_img is None:
            logger.error(f"Không thể đọc ảnh từ {left_path} hoặc {right_path}")
            return None
        
        # Tiền xử lý (cải tiến thêm kernel size adaptive)
        kernel_size = max(3, min(left_img.shape) // 20 | 1)  # Đảm bảo số lẻ
        left_img = self.preprocess_image(left_img, kernel_size)
        right_img = self.preprocess_image(right_img, kernel_size)
        
        # Phát hiện vùng quan tâm (ROI) với contour tối ưu hơn
        left_roi = self.detect_retinal_region(left_img)
        right_roi = self.detect_retinal_region(right_img)
        
        # Kiểm tra diện tích ROI hợp lệ
        if np.sum(left_roi) == 0 or np.sum(right_roi) == 0:
            logger.warning(f"Không phát hiện được võng mạc cho cặp {pair_id}")
            return None
        
        # Ước lượng độ cong với làm mờ Gaussian thích ứng
        left_curvature = self.estimate_curvature(left_roi)
        right_curvature = self.estimate_curvature(right_roi)
        
        # Tính toán depth map với hàm chuyển đổi phi tuyến
        left_depth = self.calculate_depth_from_curvature(left_curvature)
        right_depth = self.calculate_depth_from_curvature(right_curvature)
        
        # Kết hợp depth map với trọng số động
        combined_depth = self.combine_depths(left_depth, right_depth, left_roi, right_roi)
        
        # Lưu kết quả
        output_path = output_dir / f"{pair_id}_depth_map.jpg"
        cv2.imwrite(str(output_path), combined_depth)
        
        # Phân tích và lưu thông số
        stats = self.analyze_depth_map(combined_depth, output_path.name)
        self.analysis_results.append(stats)
        
        # Hiển thị mẫu
        if len(self.analysis_results) <= self.display_samples:
            self.display_depth_analysis(combined_depth, stats)
        
        self.processed_pairs.add(pair_id)
        logger.info(f"Đã xử lý và lưu: {output_path}")
        return combined_depth
    
    def preprocess_image(self, img, kernel_size=5):
        """Tiền xử lý ảnh cải tiến"""
        # Làm mờ Gaussian với sigma động
        sigma = max(1, kernel_size // 3)
        img = gaussian_filter(img, sigma=sigma)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # Ngưỡng thích ứng với kernel size động
        img = cv2.adaptiveThreshold(img, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, kernel_size, 2)
        return img
    
    def detect_retinal_region(self, img):
        """Phát hiện vùng võng mạc cải tiến"""
        # Tìm contours với method chính xác hơn
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros_like(img)
        
        # Lọc contour theo diện tích
        min_area = img.size * 0.1  # Ít nhất 10% diện tích ảnh
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if not valid_contours:
            return np.zeros_like(img)
        
        # Tạo mask từ contour lớn nhất
        largest_contour = max(valid_contours, key=cv2.contourArea)
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
        
        return mask
    
    def estimate_curvature(self, roi):
        """Ước lượng độ cong cải tiến"""
        moments = cv2.moments(roi)
        if moments["m00"] == 0:
            return np.zeros_like(roi, dtype=np.float32)
            
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        
        y, x = np.indices(roi.shape)
        x = x - cx
        y = y - cy
        
        radius = np.sqrt(x**2 + y**2)
        radius[roi == 0] = 0
        
        # Chuẩn hóa với epsilon để tránh chia 0
        max_radius = np.max(radius) + 1e-10
        radius = radius / max_radius
        
        # Áp dụng hàm mũ để nhấn mạnh vùng trung tâm
        curvature = np.exp(-radius**2 / 0.2)
        return curvature
    
    def calculate_depth_from_curvature(self, curvature_map):
        """Tính depth map với gamma correction"""
        depth = 1.0 - curvature_map
        depth = np.clip(depth, 0, 1)
        
        # Gamma correction để tăng cường chi tiết
        gamma = 0.8
        depth = np.power(depth, gamma)
        
        # Chuẩn hóa và chuyển đổi
        depth = (depth * 255).astype(np.uint8)
        return depth
    
    def combine_depths(self, left_depth, right_depth, left_roi, right_roi):
        """Kết hợp depth map với trọng số theo chất lượng ROI"""
        # Tính trọng số dựa trên diện tích ROI
        left_weight = np.sum(left_roi) / (np.sum(left_roi) + np.sum(right_roi) + 1e-10)
        right_weight = 1 - left_weight
        
        combined = cv2.addWeighted(left_depth, left_weight, 
                                 right_depth, right_weight, 0)
        return combined
    
    def analyze_depth_map(self, depth_map, filename):
        """Phân tích các đại lượng của depth map"""
        valid_pixels = depth_map[depth_map > 0]
        
        if len(valid_pixels) == 0:
            return {
                "File": filename,
                "Min": 0,
                "Max": 0,
                "Mean": 0,
                "Median": 0,
                "Std": 0,
                "Range": 0,
                "Non-zero (%)": 0,
                "Entropy": 0
            }
        
        # Tính entropy
        hist = np.histogram(valid_pixels, bins=256, range=(0,255))[0]
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return {
            "File": filename,
            "Min": np.min(valid_pixels),
            "Max": np.max(valid_pixels),
            "Mean": np.mean(valid_pixels),
            "Median": np.median(valid_pixels),
            "Std": np.std(valid_pixels),
            "Range": np.ptp(valid_pixels),
            "Non-zero (%)": len(valid_pixels) / depth_map.size * 100,
            "Entropy": entropy
        }
    
    def display_depth_analysis(self, depth_map, stats):
        """Hiển thị phân tích depth map"""
        plt.figure(figsize=(15, 5))
        
        # Depth map
        plt.subplot(1, 3, 1)
        plt.imshow(depth_map, cmap='viridis')
        plt.title(f"Depth Map\n{stats['File']}")
        plt.colorbar()
        
        # Histogram
        plt.subplot(1, 3, 2)
        plt.hist(depth_map.flatten(), bins=50, range=(0, 255))
        plt.title("Histogram Giá Trị Độ Sâu")
        plt.xlabel("Giá trị độ sâu")
        plt.ylabel("Số lượng pixel")
        
        # Thông số
        plt.subplot(1, 3, 3)
        text = "\n".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" 
                         for k, v in stats.items()])
        plt.text(0.1, 0.5, text, fontsize=10, va='center')
        plt.axis('off')
        plt.title("Thông Số Thống Kê")
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self):
        """Tạo báo cáo tổng hợp"""
        if not self.analysis_results:
            return None
            
        df = pd.DataFrame(self.analysis_results)
        
        print("\nBÁO CÁO TỔNG HỢP DEPTH MAPS")
        print("="*60)
        print(df.describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])
        
        return df

def process_all_pairs():
    """Xử lý tất cả các cặp ảnh trong thư mục"""
    estimator = RetinalDepthEstimator()
    pair_dir = config.PAIR_PREPROCESSED_DIR
    output_dir = config.DEPTH_MAPS_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    left_images = sorted(pair_dir.glob("*_left.jpg"))
    right_images = {img.name.replace("_left.jpg", "_right.jpg"): img 
                   for img in pair_dir.glob("*_right.jpg")}
    
    print(f"\nBắt đầu xử lý {len(left_images)} ảnh left...")
    print(f"Thư mục đầu ra: {output_dir}")
    
    processed_count = 0
    for i, left_path in enumerate(left_images, 1):
        right_name = left_path.name.replace("_left.jpg", "_right.jpg")
        right_path = pair_dir / right_name
        
        if right_path.exists():
            try:
                depth_map = estimator.process_pair(left_path, right_path, output_dir)
                if depth_map is not None:
                    processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"Đã xử lý {processed_count}/{len(left_images)} cặp...")
                    
            except Exception as e:
                logger.error(f"Lỗi khi xử lý {left_path}: {str(e)}")
        else:
            logger.warning(f"Không tìm thấy ảnh right tương ứng cho {left_path}")
    
    # Xuất báo cáo tổng hợp
    print(f"\nHoàn thành! Đã xử lý {processed_count} cặp ảnh")
    estimator.generate_summary_report()
    print(f"Kết quả được lưu tại: {output_dir}")

if __name__ == "__main__":
    process_all_pairs()