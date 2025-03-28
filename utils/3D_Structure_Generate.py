import open3d as o3d
import numpy as np

def create_coordinate_axes():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([
        [0, 0, 0], [0.02, 0, 0],
        [0, 0, 0], [0, 0.02, 0],
        [0, 0, 0], [0, 0, 0.02]
    ])
    axes.lines = o3d.utility.Vector2iVector([
        [0, 1], [2, 3], [4, 5]
    ])
    axes.colors = o3d.utility.Vector3dVector([
        [1, 0, 0], [0, 1, 0], [0, 0, 1]
    ])
    return axes

def create_eye_wireframe():
    eye_radius = 0.012
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=eye_radius)
    sphere.compute_vertex_normals()
    
    vertices = np.asarray(sphere.vertices)
    triangles = np.asarray(sphere.triangles)
    
    cornea_vertices = []
    k = 0.001 
    
    for i, v in enumerate(vertices):
        if v[2] > eye_radius * 0.8:
            r = np.sqrt(v[0]**2 + v[1]**2)
            bulge = k * (1 - (r**2 / eye_radius**2)) 
            vertices[i] += np.array([0, 0, bulge])
            cornea_vertices.append(vertices[i])
    
    sphere.vertices = o3d.utility.Vector3dVector(vertices)
    wireframe_eye = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    wireframe_eye.paint_uniform_color([1, 1, 1])
    
    cornea = o3d.geometry.TriangleMesh()
    cornea.vertices = o3d.utility.Vector3dVector(np.array(cornea_vertices))
    cornea.triangles = o3d.utility.Vector3iVector(triangles)
    wireframe_cornea = o3d.geometry.LineSet.create_from_triangle_mesh(cornea)
    wireframe_cornea.paint_uniform_color([1, 0, 0])
    
    coordinate_axes = create_coordinate_axes()
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Eye Wireframe", width=800, height=600)
    vis.add_geometry(wireframe_eye)
    vis.add_geometry(wireframe_cornea)
    vis.add_geometry(coordinate_axes)
    
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    create_eye_wireframe()
