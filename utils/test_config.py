from utils.config import config
from pathlib import Path
import hashlib
from collections import defaultdict

def check_for_duplicates():
    print("\n=== PATH VERIFICATION ===")
    print(f"Config directory path: {config.PAIR_PREPROCESSED_DIR}")
    print(f"Scanning directory: {config.PAIR_PREPROCESSED_DIR.absolute()}")
    print(f"Directory exists: {config.PAIR_PREPROCESSED_DIR.exists()}")
    
    # Debug: List all files in the directory
    all_files = list(config.PAIR_PREPROCESSED_DIR.glob("*"))
    print(f"\nAll files in directory: {[f.name for f in all_files]}")

    if not all_files:
        print("\nThe directory is empty or no files are accessible.")
        return

    # First show what files are actually present
    sample_files = list(config.PAIR_PREPROCESSED_DIR.glob("*"))
    print(f"\nFound {len(sample_files)} total items")
    print("Sample files (first 5):")
    for f in sample_files[:5]:
        print(f"- {f.name}")

    print("\n=== DUPLICATE DETECTION ===")
    hash_groups = defaultdict(list)
    valid_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']
    
    # Group files by their base name (e.g., "0" for "0_left.jpg" and "0_right.jpg")
    paired_files = defaultdict(list)
    for img_path in config.PAIR_PREPROCESSED_DIR.glob("*"):
        if img_path.suffix in valid_extensions:
            base_name = "_".join(img_path.stem.split("_")[:-1])  # Extract base name (e.g., "0" from "0_left")
            paired_files[base_name].append(img_path)

    # Debug: Print grouped pairs
    print("\nGrouped pairs:")
    for base_name, files in paired_files.items():
        print(f"{base_name}: {[f.name for f in files]}")

    # Hash each pair of images
    for base_name, files in paired_files.items():
        if len(files) == 2:  # Only process pairs
            try:
                combined_hash = hashlib.md5()
                for file in sorted(files):  # Ensure consistent order
                    with open(file, "rb") as f:
                        combined_hash.update(f.read())
                hash_groups[combined_hash.hexdigest()].append([f.name for f in files])
            except Exception as e:
                print(f"Error processing pair {base_name}: {str(e)}")

    duplicates = {k: v for k, v in hash_groups.items() if len(v) > 1}
    
    print("\n=== RESULTS ===")
    print(f"Valid pairs found: {sum(len(v) for v in hash_groups.values())}")
    print(f"Unique pairs: {len(hash_groups)}")
    
    if duplicates:
        print(f"\nDuplicate sets: {len(duplicates)}")
        for hash_val, pairs in list(duplicates.items())[:3]:
            print(f"\nMD5: {hash_val[:8]}...")
            for pair in pairs[:3]:
                print(f"- Pair: {pair}")
    else:
        print("\nNo duplicates found")

if __name__ == "__main__":
    config.ensure_directories()
    check_for_duplicates()