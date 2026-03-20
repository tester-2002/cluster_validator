import hashlib
import sys
import os
from pathlib import Path

def calculate_md5(file_path):
    """Calculate MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

if __name__ == "__main__":
    # If no arguments provided, process all files in current directory
    if len(sys.argv) < 2:
        files_to_process = sorted([f for f in os.listdir('.') if os.path.isfile(f)])
        if not files_to_process:
            print("No files found in current directory")
            sys.exit(1)
    else:
        files_to_process = sys.argv[1:]
    
    # Process all files
    for file_path in files_to_process:
        checksum = calculate_md5(file_path)
        print(f"{checksum}  {file_path}")