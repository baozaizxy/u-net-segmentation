import os
import shutil

def move_images(source_dir):
    target_dir = os.path.join(source_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    valid_extensions = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG',)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(valid_extensions):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, file)
                shutil.move(src_path, dst_path)
                print(f"Moved: {src_path} to {dst_path}")

source_directory = './dataset/valid/'
move_images(source_directory)
