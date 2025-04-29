import torch
from torch.utils.cpp_extension import load
import os
import subprocess
import sys

def build():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    setup_path = os.path.join(script_dir, 'setup.py')
    
    # Ensure pip is available
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', '--version'])
    except subprocess.CalledProcessError:
        print("Error: pip is not available. Please ensure pip is installed.")
        sys.exit(1)

    # Clean previous builds first
    print("Cleaning previous build directories...")
    build_dir = os.path.join(script_dir, 'build')
    egg_info_dir = os.path.join(script_dir, 'cuda_kdon.egg-info')
    
    try:
        subprocess.check_call([sys.executable, setup_path, 'clean', '--all'], cwd=script_dir)
        # Manually remove directories if 'clean --all' doesn't catch them
        if os.path.exists(build_dir):
            import shutil
            print(f"Removing {build_dir}...")
            shutil.rmtree(build_dir)
        if os.path.exists(egg_info_dir):
            import shutil
            print(f"Removing {egg_info_dir}...")
            shutil.rmtree(egg_info_dir)
        print("Clean completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error during clean: {e}")
        # Continue building even if clean fails

    print("Building KDON extension...")
    try:
        subprocess.check_call([
            sys.executable, setup_path, 'install'
        ], cwd=script_dir)
        print("KDON extension built and installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error building KDON extension: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build() 