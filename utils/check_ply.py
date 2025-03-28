import open3d as o3d
import os
from pathlib import Path
from utils.config import config
from utils.logger import logger

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    encoding="utf-8"  # Thêm encoding UTF-8
)

def check_ply_files(models_dir, output_dir, num_files=10):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ply_check_results.txt"

    ply_files = sorted(models_dir.glob("*.ply"))
    if not ply_files:
        logger.error(f"Không tìm thấy file .ply nào trong {models_dir}")
        return

    ply_files = ply_files[:min(num_files, len(ply_files))]
    logger.info(f"Đang kiểm tra {len(ply_files)} file .ply")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Kết quả kiểm tra file .ply\n")
        f.write("=" * 50 + "\n")

        for ply_path in ply_files:
            file_name = ply_path.name
            logger.info(f"Kiểm tra file: {file_name}")
            try:
                mesh = o3d.io.read_triangle_mesh(str(ply_path))
                num_vertices = len(mesh.vertices)
                num_triangles = len(mesh.triangles)
                has_vertex_normals = mesh.has_vertex_normals()
                has_triangle_normals = mesh.has_triangle_normals()

                f.write(f"File: {file_name}\n")
                f.write(f"- Số đỉnh (vertices): {num_vertices}\n")
                f.write(f"- Số tam giác (triangles): {num_triangles}\n")
                f.write(f"- Có pháp tuyến đỉnh: {has_vertex_normals}\n")
                f.write(f"- Có pháp tuyến tam giác: {has_triangle_normals}\n")
                
                if num_vertices == 0:
                    status = "Lỗi: Không có đỉnh"
                elif num_triangles == 0:
                    status = "Cảnh báo: Chỉ có đám mây điểm, không có lưới tam giác"
                elif num_vertices < 100 or num_triangles < 50:
                    status = "Cảnh báo: Mô hình quá thưa"
                else:
                    status = "OK: Mô hình hợp lệ"
                f.write(f"- Trạng thái: {status}\n")
                f.write("-" * 50 + "\n")

                logger.info(f"{file_name} - Vertices: {num_vertices}, Triangles: {num_triangles}, Status: {status}")

            except Exception as e:
                error_msg = f"Lỗi khi đọc file {file_name}: {str(e)}"
                logger.error(error_msg)
                f.write(f"File: {file_name}\n")
                f.write(f"- {error_msg}\n")
                f.write("-" * 50 + "\n")

    logger.info(f"Kết quả kiểm tra đã được lưu vào {output_file}")

if __name__ == "__main__":
    models_dir = config.MODELS_3D_DIR
    output_dir = Path("D:/3D_Stimualation/output")  # Đổi nếu cần
    check_ply_files(models_dir, output_dir, num_files=10)