"""Package source code into a zip for Kaggle Dataset upload.

Usage:
    python package_for_kaggle.py

This creates `llm-from-scratch-code.zip` containing all .py files needed
to run training on Kaggle without git cloning.

Upload instructions:
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload the generated zip file
4. Name it `llm-from-scratch` (or whatever you prefer)
5. In your Kaggle notebook, attach this dataset as an Input
"""
import os
import zipfile
import shutil


def package_for_kaggle():
    """Create a zip of all source files for Kaggle Dataset upload."""
    output_zip = "llm-from-scratch-code.zip"
    temp_dir = "__kaggle_package_temp"

    # Clean up any previous temp dir / zip
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    if os.path.exists(output_zip):
        os.remove(output_zip)

    os.makedirs(temp_dir, exist_ok=True)

    # Files to include at root level
    root_files = [
        "config.py",
        "prepare_data.py",
        "train.py",
        "generate.py",
        "requirements.txt",
    ]

    # Directories to include (with all .py files)
    include_dirs = [
        "data",
        "model",
        "utils",
        "tests",
    ]

    # Copy root files
    for fname in root_files:
        if os.path.exists(fname):
            shutil.copy2(fname, temp_dir)
            print(f"  Added: {fname}")
        else:
            print(f"  WARNING: {fname} not found, skipping")

    # Copy directories
    for dname in include_dirs:
        if os.path.isdir(dname):
            dst = os.path.join(temp_dir, dname)
            shutil.copytree(dname, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            print(f"  Added: {dname}/")
        else:
            print(f"  WARNING: {dname}/ not found, skipping")

    # Create zip
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(temp_dir):
            for f in files:
                filepath = os.path.join(root, f)
                arcname = os.path.relpath(filepath, temp_dir)
                zf.write(filepath, arcname)

    # Clean up temp dir
    shutil.rmtree(temp_dir)

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\n✅ Created: {output_zip} ({size_mb:.2f} MB)")
    print("\nNext steps:")
    print("  1. Go to https://www.kaggle.com/datasets")
    print("  2. Click 'New Dataset'")
    print("  3. Upload the zip above")
    print("  4. Name it 'llm-from-scratch' (or your preferred name)")
    print("  5. In your Kaggle notebook → Add Data → attach this dataset")


if __name__ == "__main__":
    package_for_kaggle()
