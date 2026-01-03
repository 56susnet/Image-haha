import json
import os

# Files that use the hash-based structure {"data": {"hash": {...}}, "default": {...}}
hash_files = [
    "scripts/lrs/flux.json",
    "scripts/lrs/person_config.json",
    "scripts/lrs/style_config.json"
]

def clean_hash_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        with open(file_path, "r") as f:
            content = json.load(f)
        
        if "data" in content:
            for key in content["data"]:
                content["data"][key] = {}
        
        content["default"] = {}

        with open(file_path, "w") as f:
            json.dump(content, f, indent=4)
        print(f"Cleaned hash-based file: {file_path}")
    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")

# Execute cleaning
for f in hash_files:
    clean_hash_file(f)
