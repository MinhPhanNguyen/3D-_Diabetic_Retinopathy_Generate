# src/utils/visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from typing import Union, Optional, List, Tuple
from pathlib import Path
from utils.config import config
from utils.logger import logger

class MeshVisualizer:
    """Class dedicated to 3D mesh visualization and analysis"""
    
    @staticmethod
    def visualize_mesh(mesh_path: Union[str, Path], 
                      window_name: str = "3D Mesh",
                      screenshot_path: Optional[Union[str, Path]] = None) -> None:
        """
        Visualize a 3D mesh with interactive controls
        
        Args:
            mesh_path: Path to mesh file (.ply, .obj, .stl)
            window_name: Title for visualization window
            screenshot_path: Optional path to save screenshot
        """
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
            
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if len(mesh.vertices) == 0:
            logger.warning(f"Empty mesh loaded from {mesh_path}")
            return
            
        # Compute normals for better visualization
        mesh.compute_vertex_normals()
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1200, height=800)
        vis.add_geometry(mesh)
        
        # Set camera to frontal view
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.8)
        
        # Save screenshot if requested
        if screenshot_path:
            vis.capture_screen_image(str(screenshot_path))
            logger.info(f"Screenshot saved to {screenshot_path}")
            
        vis.run()
        vis.destroy_window()

    @staticmethod
    def visualize_mesh_comparison(mesh_paths: List[Union[str, Path]], 
                                 window_name: str = "Mesh Comparison",
                                 titles: Optional[List[str]] = None) -> None:
        """
        Compare multiple meshes side by side
        
        Args:
            mesh_paths: List of paths to mesh files
            window_name: Title for visualization window
            titles: Optional list of titles for each mesh
        """
        if not mesh_paths:
            raise ValueError("No mesh paths provided")
            
        # Load all meshes
        meshes = []
        for path in mesh_paths:
            path = Path(path)
            if not path.exists():
                logger.warning(f"Mesh file not found: {path}")
                continue
                
            mesh = o3d.io.read_triangle_mesh(str(path))
            if len(mesh.vertices) > 0:
                mesh.compute_vertex_normals()
                meshes.append(mesh)
            else:
                logger.warning(f"Empty mesh loaded from {path}")
        
        if not meshes:
            logger.error("No valid meshes to display")
            return
            
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1600, height=900)
        
        # Add each mesh with optional title
        for i, mesh in enumerate(meshes):
            vis.add_geometry(mesh, reset_bounding_box=(i==0))
            
        # Set camera to frontal view
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, 1, 0])
        ctr.set_zoom(0.6)
        
        vis.run()
        vis.destroy_window()

    @staticmethod
    def visualize_point_cloud_with_normals(pcd_path: Union[str, Path],
                                          window_name: str = "Point Cloud") -> None:
        """
        Visualize point cloud with normal vectors
        
        Args:
            pcd_path: Path to point cloud file
            window_name: Title for visualization window
        """
        pcd_path = Path(pcd_path)
        if not pcd_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {pcd_path}")
            
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        if len(pcd.points) == 0:
            logger.warning(f"Empty point cloud loaded from {pcd_path}")
            return
            
        # Compute normals if not present
        if not pcd.has_normals():
            pcd.estimate_normals()
            
        # Create coordinate frame for reference
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        o3d.visualization.draw_geometries(
            [pcd, coord_frame],
            window_name=window_name,
            width=1200,
            height=800,
            point_show_normal=True
        )

    @staticmethod
    def generate_mesh_screenshots(mesh_dir: Union[str, Path],
                                 output_dir: Union[str, Path],
                                 views: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None):
        """
        Generate standardized screenshots of meshes from multiple views
        
        Args:
            mesh_dir: Directory containing mesh files
            output_dir: Directory to save screenshots
            views: List of (front_vector, up_vector) tuples defining camera views
        """
        mesh_dir = Path(mesh_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default views if not provided
        if views is None:
            views = [
                ((0, 0, -1), (0, 1, 0)),   # Front view
                ((-1, 0, 0), (0, 1, 0)),    # Left view
                ((1, 0, 0), (0, 1, 0)),     # Right view
                ((0, -1, 0), (0, 0, 1)),    # Top view
            ]
        
        for mesh_file in mesh_dir.glob("*.ply"):
            try:
                mesh = o3d.io.read_triangle_mesh(str(mesh_file))
                if len(mesh.vertices) == 0:
                    continue
                    
                mesh.compute_vertex_normals()
                
                for i, (front, up) in enumerate(views):
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(width=800, height=600, visible=False)
                    vis.add_geometry(mesh)
                    
                    ctr = vis.get_view_control()
                    ctr.set_front(front)
                    ctr.set_up(up)
                    ctr.set_zoom(0.8)
                    
                    screenshot_path = output_dir / f"{mesh_file.stem}_view_{i}.png"
                    vis.capture_screen_image(str(screenshot_path))
                    vis.destroy_window()
                    
                    logger.info(f"Saved view {i} for {mesh_file.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {mesh_file.name}: {str(e)}")

def visualize_mesh(mesh_path: Union[str, Path], **kwargs):
    """Convenience function for quick mesh visualization"""
    MeshVisualizer.visualize_mesh(mesh_path, **kwargs)

if __name__ == "__main__":
    # Example usage
    config.ensure_directories()
    
    # Visualize a single mesh
    test_mesh = config.MODELS_3D_DIR / "0_left.ply"
    if test_mesh.exists():
        visualize_mesh(test_mesh)
    
    # Generate screenshots from multiple views
    MeshVisualizer.generate_mesh_screenshots(
        config.MODELS_3D_DIR,
        config.VISUALIZATIONS_DIR / "mesh_screenshots"
    )