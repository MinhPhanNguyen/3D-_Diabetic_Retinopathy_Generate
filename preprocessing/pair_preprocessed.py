import shutil
from pathlib import Path
from utils.config import config

def process_paired_images():
    # Tạo thư mục đích nếu chưa tồn tại
    config.PAIR_PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Tìm tất cả các file JPG trong thư mục nguồn
    valid_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']
    all_files = [f for f in config.PREPROCESSED_DIR.glob("*") if f.suffix in valid_extensions]
    
    # Nhóm các file theo base name (phần tên trước _left/_right)
    paired_files = {}
    for img_path in all_files:
        parts = img_path.stem.split('_')
        if len(parts) >= 2:  # Đảm bảo có dạng [number]_[left/right]
            base_name = '_'.join(parts[:-1])  # Lấy phần base name
            side = parts[-1].lower()  # left hoặc right
            
            if base_name not in paired_files:
                paired_files[base_name] = {'left': None, 'right': None}
            
            paired_files[base_name][side] = img_path
    
    # Xử lý từng cặp
    for base_name, files in paired_files.items():
        left_img = files['left']
        right_img = files['right']
        
        # Chỉ xử lý nếu có cả 2 ảnh
        if left_img and right_img:
            # Tạo tên file mới
            new_left_name = f"{base_name}_left{left_img.suffix}"
            new_right_name = f"{base_name}_right{right_img.suffix}"
            
            # Đường dẫn đích
            dest_left = config.PAIR_PREPROCESSED_DIR / new_left_name
            dest_right = config.PAIR_PREPROCESSED_DIR / new_right_name
            
            # Copy file sang thư mục mới
            shutil.copy2(left_img, dest_left)
            shutil.copy2(right_img, dest_right)
    
    # Thống kê kết quả
    total_pairs = sum(1 for files in paired_files.values() if files['left'] and files['right'])
    single_images = sum(1 for files in paired_files.values() if not (files['left'] and files['right']))
    
    print(f"Đã xử lý xong!")
    print(f"- Tổng số cặp hình ảnh hợp lệ: {total_pairs}")
    print(f"- Số hình ảnh không có đôi: {single_images}")
    print(f"- File đã được lưu vào: {config.PAIR_PREPROCESSED_DIR}")

if __name__ == "__main__":
    process_paired_images()