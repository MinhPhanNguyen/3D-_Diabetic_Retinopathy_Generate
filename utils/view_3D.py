import open3d as o3d

# Đường dẫn đến file .obj
obj_path = r"D:\3D_Stimualation\data\3D_models\obj_textured\textured_1010.obj"

# Đọc và hiển thị
mesh = o3d.io.read_triangle_mesh(obj_path)
o3d.visualization.draw_geometries([mesh], window_name="Textured Mesh 1006")