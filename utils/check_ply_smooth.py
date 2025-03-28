import open3d as o3d
from pathlib import Path
import logging

# Cấu hình logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="D:/3D_Stimualation/output/logs/check_ply.log",
    filemode="w",
    encoding="utf-8"
)
logger = logging.getLogger("PlyChecker")

def check_ply_file(ply_path):
    try:
        # Đọc file .ply
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        
        # Thu thập thông tin
        num_vertices = len(mesh.vertices)
        num_triangles = len(mesh.triangles)
        has_vertex_normals = mesh.has_vertex_normals()
        has_texture = mesh.has_textures() and len(mesh.textures) > 0
        
        # Tạo chuỗi kết quả
        result = (
            f"File: {ply_path.name}\n"
            f"- Số đỉnh (vertices): {num_vertices}\n"
            f"- Số tam giác (triangles): {num_triangles}\n"
            f"- Có pháp tuyến đỉnh: {has_vertex_normals}\n"
            f"- Có texture: {has_texture}\n"
            f"- Trạng thái: {'OK' if num_triangles > 0 else 'No triangles'}\n"
        )
        logger.info(result)
        print(result)
        
        return result  # Trả về chuỗi kết quả để ghi vào file
    except Exception as e:
        error_msg = f"Error checking {ply_path}: {str(e)}"
        logger.error(error_msg)
        print(error_msg)
        return None

def check_all_ply_files(base_dir):
    # Kiểm tra thư mục refined và textured
    refined_dir = Path(base_dir) / "refined"
    textured_dir = Path(base_dir) / "textured"
    
    # Tạo file kết quả
    result_file = Path("D:/3D_Stimualation/output/ply_check_results.txt")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("Kiểm tra file .ply\n\n")
        
        # Kiểm tra thư mục refined
        logger.info("Checking refined directory")
        print("\nChecking refined directory:")
        for ply_path in refined_dir.glob("*.ply"):
            result = check_ply_file(ply_path)
            if result:
                f.write(result + "\n")
        
        # Kiểm tra thư mục textured
        logger.info("Checking textured directory")
        print("\nChecking textured directory:")
        for ply_path in textured_dir.glob("*.ply"):
            result = check_ply_file(ply_path)
            if result:
                f.write(result + "\n")
    
    logger.info("Check completed")
    print(f"\nKết quả đã được lưu vào {result_file}")

if __name__ == "__main__":
    base_dir = "D:/3D_Stimualation/data/3D_models"
    check_all_ply_files(base_dir)