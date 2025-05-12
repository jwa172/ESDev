import os
import glob

def find_latest_csv_recursive(base_dir):
    csv_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return max(csv_files, key=os.path.getmtime) if csv_files else None
