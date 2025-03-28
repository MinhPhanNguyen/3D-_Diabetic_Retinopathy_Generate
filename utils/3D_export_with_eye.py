import open3d as o3d
import numpy as np
from pathlib import Path
import logging

# Cấu hình logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="D:/3D_Stimualation/output/logs/export_with_eye.log",
    filemode="w",
    encoding="utf-8"
)
logger = logging.getLogger("3DExporterWithEye")

def create_coordinate_axes():
    """Tạo trục tọa độ (X đỏ, Y xanh lá, Z xanh dương)"""
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([
        [0, 0, 0], [0.02, 0, 0],  # X-axis
        [0, 0, 0], [0, 0.02, 0],  # Y-axis
        [0, 0, 0], [0, 0, 0.02]   # Z-axis
    ])
    axes.lines = o3d.utility.Vector2iVector([
        [0, 1], [2, 3], [4, 5]
    ])
    axes.colors = o3d.utility.Vector3dVector([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]  # Red, Green, Blue
    ])
    return axes

def create_eye_wireframe():
    """Tạo lưới mắt với bán kính 12mm và giác mạc nhô lên"""
    eye_radius = 0.012  # Bán kính mắt 12mm
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=eye_radius)
    sphere.compute_vertex_normals()
    
    vertices = np.asarray(sphere.vertices)
    triangles = np.asarray(sphere.triangles)
    
    cornea_vertices = []
    k = 0.001  # Độ nhô của giác mạc
    
    # Điều chỉnh đỉnh để tạo giác mạc
    for i, v in enumerate(vertices):
        if v[2] > eye_radius * 0.8:  # Phần trước của mắt
            r = np.sqrt(v[0]**2 + v[1]**2)
            bulge = k * (1 - (r**2 / eye_radius**2))
            vertices[i] += np.array([0, 0, bulge])
            cornea_vertices.append(vertices[i])
    
    sphere.vertices = o3d.utility.Vector3dVector(vertices)
    wireframe_eye = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    wireframe_eye.paint_uniform_color([1, 1, 1])  # Màu trắng cho toàn bộ mắt
    
    cornea = o3d.geometry.TriangleMesh()
    cornea.vertices = o3d.utility.Vector3dVector(np.array(cornea_vertices))
    cornea.triangles = o3d.utility.Vector3iVector(triangles)
    wireframe_cornea = o3d.geometry.LineSet.create_from_triangle_mesh(cornea)
    wireframe_cornea.paint_uniform_color([1, 0, 0])  # Màu đỏ cho giác mạc
    
    return wireframe_eye, wireframe_cornea

def export_to_obj(ply_path, obj_path):
    """Xuất file .ply sang .obj và trả về mesh để hiển thị"""
    try:
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        if len(mesh.triangles) == 0:
            raise ValueError("Mesh has no triangles")

        obj_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(
            str(obj_path),
            mesh,
            write_vertex_normals=True,
            write_triangle_uvs=True
        )
        logger.info(f"Exported {ply_path.name} to {obj_path}")
        print(f"Exported {ply_path.name} to {obj_path}")
        return mesh
    except Exception as e:
        logger.error(f"Failed to export {ply_path}: {str(e)}")
        print(f"Error exporting {ply_path}: {str(e)}")
        return None

def export_and_visualize(base_dir):
    """Xuất .obj và hiển thị tất cả mô hình cùng lưới mắt"""
    refined_dir = Path(base_dir) / "refined"
    textured_dir = Path(base_dir) / "textured"
    refined_obj_dir = Path(base_dir) / "obj_refined"
    textured_obj_dir = Path(base_dir) / "obj_textured"

    # Tạo lưới mắt và trục tọa độ
    wireframe_eye, wireframe_cornea = create_eye_wireframe()
    coordinate_axes = create_coordinate_axes()

    # Tạo visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Model with Eye Wireframe", width=800, height=600)
    vis.add_geometry(wireframe_eye)
    vis.add_geometry(wireframe_cornea)
    vis.add_geometry(coordinate_axes)

    # Xuất và hiển thị từ refined
    logger.info("Exporting and visualizing refined directory")
    print("\nExporting and visualizing refined directory:")
    for ply_path in refined_dir.glob("*.ply"):
        obj_path = refined_obj_dir / f"{ply_path.stem}.obj"
        mesh = export_to_obj(ply_path, obj_path)
        if mesh:
            vis.add_geometry(mesh)

    # Xuất và hiển thị từ textured
    logger.info("Exporting and visualizing textured directory")
    print("\nExporting and visualizing textured directory:")
    for ply_path in textured_dir.glob("*.ply"):
        obj_path = textured_obj_dir / f"{ply_path.stem}.obj"
        mesh = export_to_obj(ply_path, obj_path)
        if mesh:
            vis.add_geometry(mesh)

    # Cấu hình render
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])  # Nền đen
    opt.show_coordinate_frame = False  # Tắt trục mặc định của Open3D (dùng trục tự tạo)

    # Chạy visualizer
    vis.run()
    vis.destroy_window()

    logger.info("Export and visualization completed")
    print("\nExport and visualization completed")

if __name__ == "__main__":
    base_dir = "D:/3D_Stimualation/data/3D_models"
    export_and_visualize(base_dir)