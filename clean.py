#!/usr/bin/env python3
"""
Script to clean generated files from the build process.
"""

import os
import shutil
import glob


def clean():
    root_dir = os.path.dirname(os.path.abspath(__file__))

    patterns_to_remove = [
        "build/",
        "**/*.so",
        "**/*.o",
        
        "**/*.html",
        "**/*.cpp",

        "**/__pycache__/",
    ]
    
    protected_files = [
        os.path.join(root_dir, "include/SpatialBirthDeath.h"),
        os.path.join(root_dir, "src/SpatialBirthDeath.cpp"),
        os.path.join(root_dir, "simulation/SpatialBirthDeathWrapper.pyx"),
    ]
    
    for pattern in patterns_to_remove:
        for item in glob.glob(os.path.join(root_dir, pattern), recursive=True):

            if item in protected_files:
                continue

            if "env" in item: # check for ven/ or env/ in path
                continue

            item = os.path.normpath(item)
            
            try:  
                if os.path.isdir(item):
                    print(f"Removing directory: {os.path.relpath(item, root_dir)}")
                    shutil.rmtree(item)
                elif os.path.isfile(item):
                    print(f"Removing file: {os.path.relpath(item, root_dir)}")
                    os.remove(item)
            except Exception as e:
                print(f"Error removing {item}: {e}")
    
    print("Cleanup complete!")


if __name__ == "__main__":
    clean()