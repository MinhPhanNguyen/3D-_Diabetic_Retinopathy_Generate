import open3d as o3d
import numpy as np
from pathlib import Path
from PIL import Image
from utils.config import config
from utils.logger import logger

class TextureMapper:
    def __init__(self):
        self.uv_atlas_resolution = 2048
        self.texture_fill_threshold = 0.1

    def apply_texture(self, mesh_path, image_path, output_path):
        try:
            # Validate inputs
            if not Path(mesh_path).exists():
                raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Read data
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if len(mesh.vertices) == 0:
                raise ValueError("Mesh contains no vertices")

            image = o3d.io.read_image(str(image_path))
            
            # Process mesh
            mesh = self._preprocess_mesh(mesh)
            mesh = self._parameterize(mesh)
            texture_atlas = self._create_atlas(mesh, image)
            
            # Apply texture with material
            mesh.textures = [texture_atlas]
            if not mesh.has_triangle_uvs():
                raise ValueError("Failed to generate triangle UVs")
            mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(len(mesh.triangles), dtype=np.int32))  # Gán material ID
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(
                str(output_path),
                mesh,
                write_vertex_normals=True,
                write_triangle_uvs=True,
                write_ascii=True  # Dễ kiểm tra file .ply
            )
            logger.info(f"Successfully textured: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Texturing failed for {mesh_path}: {str(e)}")
            return False

    def _preprocess_mesh(self, mesh):
        """Prepare mesh for texturing"""
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        return mesh

    def _parameterize(self, mesh):
        """Generate UV coordinates using Open3D's built-in method"""
        mesh.compute_triangle_uvs()  # Tạo UV tự động
        return mesh

    def _create_atlas(self, mesh, image):
        """Create texture atlas with proper dimensions"""
        img_array = np.asarray(image)
        
        # Resize to power-of-two dimensions if needed
        if not (img_array.shape[0] & (img_array.shape[0] - 1) == 0):
            new_size = 2 ** int(np.ceil(np.log2(img_array.shape[0])))
            img_array = np.array(Image.fromarray(img_array).resize((new_size, new_size)))
            
        return o3d.geometry.Image(img_array)