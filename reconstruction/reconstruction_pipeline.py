from .pointcloud_to_mesh import MeshGenerator
from .texture_mapping import TextureMapper
from .mesh_postprocess import MeshPostprocessor
from utils.config import config
from pathlib import Path
import time
from utils.logger import logger

def run_pipeline():
    logger.info("Starting 3D reconstruction pipeline")
    
    # 1. Generate raw meshes
    logger.info("Stage 1: Generating raw meshes")
    start_time = time.time()
    mesh_gen = MeshGenerator()
    
    left_images = sorted(config.PAIR_PREPROCESSED_DIR.glob("*_left.jpg"))
    depth_maps = sorted(config.DEPTH_MAPS_DIR.glob("*_depth_map.jpg"))
    depth_map_dict = {mesh_gen.extract_id(d): d for d in depth_maps}
    right_images = {mesh_gen.extract_id(p): p for p in config.PAIR_PREPROCESSED_DIR.glob("*_right.jpg")}

    processed_count = 0
    for left_path in left_images[:5]:  # Giới hạn 5 file
        pair_id = mesh_gen.extract_id(left_path)
        right_path = right_images.get(pair_id)
        depth_path = depth_map_dict.get(pair_id)
        if right_path and depth_path:
            output_path = config.MODELS_3D_DIR / f"{pair_id}.ply"
            if mesh_gen.process_pair(left_path, right_path, depth_path, output_path):
                processed_count += 1
        else:
            missing = []
            if not right_path: missing.append("right image")
            if not depth_path: missing.append("depth map")
            logger.warning(f"Skipping {pair_id}: Missing {' and '.join(missing)}")
    
    logger.info(f"Mesh generation completed in {time.time() - start_time:.2f}s, processed {processed_count} pairs")

    # 2. Post-process meshes
    logger.info("Stage 2: Post-processing meshes")
    post_processor = MeshPostprocessor()
    processed_count = 0
    
    for mesh_path in Path(config.MODELS_3D_DIR).glob("*.ply"):
        if not mesh_path.name.endswith('.initial.ply') and not mesh_path.name.endswith('.debug_normals.ply'):
            output_path = config.MODELS_3D_DIR / "refined" / f"refined_{mesh_path.name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if post_processor.process_mesh(mesh_path, output_path):
                processed_count += 1
    logger.info(f"Post-processed {processed_count} meshes")

    # 3. Apply textures
    logger.info("Stage 3: Applying textures")
    tex_mapper = TextureMapper()
    textured_count = 0
    
    for mesh_path in Path(config.MODELS_3D_DIR / "refined").glob("*.ply"):
        if mesh_path.name.startswith('refined_'):
            img_path = Path(config.PAIR_PREPROCESSED_DIR) / f"{mesh_path.stem[8:]}_left.jpg"
            output_path = config.MODELS_3D_DIR / "textured" / f"textured_{mesh_path.stem[8:]}.ply"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if tex_mapper.apply_texture(mesh_path, img_path, output_path):
                textured_count += 1
    logger.info(f"Textured {textured_count} meshes")

    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    config.ensure_directories()
    (config.MODELS_3D_DIR / "refined").mkdir(parents=True, exist_ok=True)
    (config.MODELS_3D_DIR / "textured").mkdir(parents=True, exist_ok=True)
    run_pipeline()