import os
import shutil

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset")
)

def flatten(class_name):
    class_path = os.path.join(BASE_DIR, class_name)
    print(f"Processing {class_name}...")

    for root, dirs, files in os.walk(class_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                src = os.path.join(root, file)
                dst = os.path.join(class_path, file)

                if src != dst:
                    if not os.path.exists(dst):
                        shutil.move(src, dst)

    # Remove empty folders
    for root, dirs, files in os.walk(class_path, topdown=False):
        for d in dirs:
            folder = os.path.join(root, d)
            if not os.listdir(folder):
                os.rmdir(folder)

flatten("benign")
flatten("malignant")

print("✅ Dataset flattened successfully")
