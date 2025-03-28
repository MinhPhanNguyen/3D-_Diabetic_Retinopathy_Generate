from pathlib import Path

def check_files():
    actual_path = Path(r"D:\3D_Stimualation\data\preprocessed")
    print(f"\nChecking ACTUAL path: {actual_path}")
    print(f"Directory exists: {actual_path.exists()}")
    
    # Direct file check
    test_file = actual_path / "0_left.jpg"
    print(f"\nTest file exists: {test_file.exists()}")
    print(f"Test file path: {test_file}")
    
    # Count files
    files = list(actual_path.glob("*.jpg"))
    print(f"\nFound {len(files)} JPG files")
    if files:
        print("First 5 files:")
        for f in files[:5]:
            print(f"- {f.name}")

if __name__ == "__main__":
    check_files()