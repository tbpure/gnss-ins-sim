import os

def list_immediate_subdirectories(root_dir, path_only:bool=True):
    if path_only:
        """获取指定目录下的所有直接子文件夹路径（不包含文件）"""
        subdirs = []
        with os.scandir(root_dir) as entries:
            for entry in entries:
                if entry.is_dir():  # 只处理文件夹类型
                    subdirs.append(entry.path)
        return subdirs
    else:
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
