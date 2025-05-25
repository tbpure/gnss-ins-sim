import os

def list_immediate_subdirectories(root_dir):
    return [
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name))
    ]

if __name__ == '__main__':
    # 使用示例
    root_directory = "./demo_saved_data"
    dirs = list_immediate_subdirectories(root_directory)
    for dir_path in dirs:
        print(dir_path)