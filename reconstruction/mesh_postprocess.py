import open3d as o3d
import numpy as np
from pathlib import Path
from utils.config import config
from utils.logger import logger

class MeshPostprocessor:
    def __init__(self):
        self.params = {
            'decimate_target': 0.6,  # Keep 60% of triangles
            'smooth_iterations': 5,
            'remove_outliers': True,
            'outlier_std_ratio': 2.5,
            'min_component_size': 100
        }

    def process_mesh(self, input_path, output_path):
        try:
            logger.info(f"Processing: {input_path.name}")
            
            # Load and validate
            mesh = o3d.io.read_triangle_mesh(str(input_path))
            if len(mesh.triangles) == 0:
                raise ValueError("Mesh has no triangles")

            # Processing pipeline
            mesh = self._clean_mesh(mesh)
            mesh = self._decimate_mesh(mesh)
            mesh = self._smooth_mesh(mesh)
            mesh = self._remove_outliers(mesh)
            mesh = self._reorient_normals(mesh)
            
            # Save result
            output_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(
                str(output_path),
                mesh,
                write_vertex_normals=True
            )
            logger.info(f"Processed: {output_path}")  # Thay logger.success thành logger.info
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {input_path}: {str(e)}")
            return False

    def _clean_mesh(self, mesh):
        """Basic mesh cleaning"""
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        return mesh

    def _decimate_mesh(self, mesh):
        """Reduce triangle count while preserving shape"""
        target_tris = max(10000, int(len(mesh.triangles) * self.params['decimate_target']))
        return mesh.simplify_quadric_decimation(target_tris)

    def _smooth_mesh(self, mesh):
        """Laplacian smoothing"""
        return mesh.filter_smooth_taubin(
            number_of_iterations=self.params['smooth_iterations']
        )

    def _remove_outliers(self, mesh):
        """Remove small disconnected components"""
        if not self.params['remove_outliers']:
            return mesh
            
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangles_to_keep = [
            cluster_n_triangles[cluster] >= self.params['min_component_size']
            for cluster in triangle_clusters
        ]
        mesh.remove_triangles_by_mask(~np.array(triangles_to_keep))
        mesh.remove_unreferenced_vertices()
        return mesh

    def _reorient_normals(self, mesh):
        """Ensure consistent normal orientation"""
        mesh.orient_triangles()
        mesh.compute_vertex_normals()
        return mesh