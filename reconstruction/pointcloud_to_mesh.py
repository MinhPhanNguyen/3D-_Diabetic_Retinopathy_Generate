import open3d as o3d
import numpy as np
import os
from pathlib import Path
import logging

# Tạo thư mục log nếu chưa tồn tại
log_dir = Path("D:/3D_Stimualation/output/logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Cấu hình logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="D:/3D_Stimualation/output/logs/run.log",
    filemode="w",
    encoding="utf-8"
)
logger = logging.getLogger("3D_Diabetic_Retinopathy")
logger.info("Starting pointcloud_to_mesh script")

try:
    from utils.config import config
except ImportError as e:
    logger.error(f"Failed to import config: {str(e)}")
    raise

class MeshGenerator:
    def __init__(self):
        self.processed_ids = set()
        self.params = {
            'depth_scale': 1000.0,
            'voxel_size': 1.0,
            'poisson_depth': 14,
            'depth_trunc': 10.0,
            'mesh_postprocess': True,
            'normal_estimation_radius': 0.1,
            'normal_estimation_max_nn': 100,
            'min_points': 10
        }
        self.intrinsic = None

    def extract_id(self, path):
        return Path(path).stem.split('_')[0]

    def _validate_images(self, color_img, depth_img):
        color_array = np.asarray(color_img)
        depth_array = np.asarray(depth_img)
        if color_array.size == 0 or depth_array.size == 0:
            raise ValueError("Empty image detected")
        if color_array.shape[:2] != depth_array.shape[:2]:
            raise ValueError("Color and depth image dimensions mismatch")
        return color_array.shape[1], color_array.shape[0]

    def _create_rgbd(self, color_img, depth_img, pair_id):
        depth_array = np.asarray(depth_img)
        logger.info(f"Depth range for {pair_id}: {depth_array.min()} to {depth_array.max()}")
        import cv2
        cv2.imwrite(f"D:/output/depth_{pair_id}.jpg", depth_array)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img,
            depth_scale=self.params['depth_scale'],
            depth_trunc=self.params['depth_trunc'],
            convert_rgb_to_intensity=False
        )
        return rgbd

    def _estimate_normals(self, pcd):
        if len(pcd.points) < self.params['min_points']:
            logger.warning(f"Insufficient points ({len(pcd.points)}) for normal estimation")
            return pcd
        logger.info(f"Estimating normals with radius={self.params['normal_estimation_radius']}, max_nn={self.params['normal_estimation_max_nn']}")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.params['normal_estimation_radius'],
                max_nn=self.params['normal_estimation_max_nn']
            )
        )
        pcd.orient_normals_consistent_tangent_plane(100)
        if not pcd.has_normals():
            logger.error("Failed to estimate normals")
            raise ValueError("Point cloud has no normals after estimation")
        return pcd

    def _postprocess_mesh(self, mesh):
        mesh.remove_duplicated_vertices()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
        mesh = mesh.filter_smooth_simple(number_of_iterations=3)
        mesh.compute_vertex_normals()
        return mesh

    def process_pair(self, left_path, right_path, depth_path, output_path, resolution='medium'):
        resolution_params = {
            'high': {'voxel_size': 0.5, 'poisson_depth': 16},
            'medium': {'voxel_size': 1.0, 'poisson_depth': 14},
            'low': {'voxel_size': 2.0, 'poisson_depth': 12}
        }
        self.params.update(resolution_params[resolution])
        pair_id = self.extract_id(left_path)
        logger.info(f"Processing pair {pair_id}")

        try:
            # 1. Read and validate images
            color_img = o3d.io.read_image(str(left_path))
            depth_img = o3d.io.read_image(str(depth_path))
            width, height = self._validate_images(color_img, depth_img)
            self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=width, height=height, fx=width, fy=height, cx=width/2, cy=height/2
            )

            # 2. Create RGBD image
            rgbd = self._create_rgbd(color_img, depth_img, pair_id)

            # 3. Create point cloud
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
            logger.info(f"Initial point cloud: {len(pcd.points)} points")
            if len(pcd.points) < self.params['min_points']:
                raise ValueError(f"Too few points in initial point cloud: {len(pcd.points)}")
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            # Save initial point cloud
            initial_pcd_path = output_path.with_suffix('.initial.ply')
            o3d.io.write_point_cloud(str(initial_pcd_path), pcd)
            logger.info(f"Saved initial point cloud to {initial_pcd_path}")

            # 4. Không downsample, tính pháp tuyến
            logger.info(f"Points before processing: {len(pcd.points)} points")
            pcd = self._estimate_normals(pcd)

            # Debug: Save point cloud with normals
            debug_pcd_path = output_path.with_suffix('.debug_normals.ply')
            o3d.io.write_point_cloud(str(debug_pcd_path), pcd)
            logger.info(f"Saved point cloud with normals to {debug_pcd_path}")

            # 5. Surface reconstruction
            logger.info("Starting Poisson surface reconstruction")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=self.params['poisson_depth'], linear_fit=True, n_threads=os.cpu_count()
            )
            logger.info(f"Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

            if len(mesh.triangles) == 0:
                logger.warning(f"Poisson failed for {pair_id}. Trying Ball Pivoting...")
                radii = [0.05, 0.1, 0.2, 0.5]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )
                logger.info(f"Ball Pivoting result: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

            if len(mesh.triangles) == 0:
                raise ValueError("Both Poisson and Ball Pivoting failed to create triangles")

            # 6. Post-processing (Đã sửa lỗi densities.size)
            if len(densities) > 0:  # Sửa từ densities.size thành len(densities)
                density_threshold = np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(densities < density_threshold)
            mesh = self._postprocess_mesh(mesh)
            logger.info(f"After post-processing: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

            # 7. Save final mesh
            output_path.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_triangle_mesh(str(output_path), mesh)
            logger.info(f"Saved final mesh to {output_path}")

            return True
        except Exception as e:
            logger.error(f"Failed to process {pair_id}: {str(e)}")
            return False

def clear_previous_ply_files(models_dir):
    ply_files = list(models_dir.glob("*.ply"))
    if not ply_files:
        logger.info(f"Không tìm thấy file .ply nào để xóa trong {models_dir}")
        return
    for ply_file in ply_files:
        try:
            os.remove(ply_file)
            logger.info(f"Đã xóa file: {ply_file}")
        except Exception as e:
            logger.error(f"Không thể xóa file {ply_file}: {str(e)}")

def process_all_pairs():
    logger.info("Starting process_all_pairs")
    generator = MeshGenerator()
    
    clear_previous_ply_files(config.MODELS_3D_DIR)

    left_images = sorted(config.PAIR_PREPROCESSED_DIR.glob("*_left.jpg"))
    depth_maps = sorted(config.DEPTH_MAPS_DIR.glob("*_depth_map.jpg"))
    depth_map_dict = {generator.extract_id(d): d for d in depth_maps}
    right_images = {generator.extract_id(p): p for p in config.PAIR_PREPROCESSED_DIR.glob("*_right.jpg")}

    logger.info(f"\nFound {len(left_images)} left images")
    logger.info(f"Found {len(right_images)} right images")
    logger.info(f"Found {len(depth_maps)} depth maps")
    logger.info(f"Depth maps found: {[d.name for d in depth_maps]}")

    processed_count = 0
    for left_path in left_images[:5]:  # Chỉ xử lý 5 file
        pair_id = generator.extract_id(left_path)
        right_path = right_images.get(pair_id)
        depth_path = depth_map_dict.get(pair_id)
        if right_path and depth_path:
            output_path = config.MODELS_3D_DIR / f"{pair_id}.ply"
            if generator.process_pair(left_path, right_path, depth_path, output_path):
                processed_count += 1
        else:
            missing = []
            if not right_path: missing.append("right image")
            if not depth_path: missing.append("depth map")
            logger.warning(f"Skipping {pair_id}: Missing {' and '.join(missing)}")

    logger.info(f"\nProcessing complete! Successfully processed {processed_count} pairs")

if __name__ == "__main__":
    try:
        config.ensure_directories()
        process_all_pairs()
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        raise