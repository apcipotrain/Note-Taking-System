import pathlib


def draw_tree(path, indent=""):
    """
    递归生成目录树结构
    """
    path = pathlib.Path(path)

    # 过滤掉一些不想显示的目录（如 __pycache__, .git 等）
    ignored = {'.git', '__pycache__', '.venv', '.vscode'}

    # 获取当前路径下的所有文件和文件夹，并排序（文件夹在前，文件在后）
    try:
        # 获取并排序：文件夹在前，文件在后
        items = sorted(list(path.iterdir()), key=lambda x: (x.is_file(), x.name))
        items = [item for item in items if item.name not in ignored]
    except PermissionError:
        print(f"{indent} [! 权限不足，无法访问]")
        return

    for i, item in enumerate(items):
        # 判断是否为当前层级的最后一个元素，决定使用 └── 还是 ├──
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "

        # 打印当前项
        print(f"{indent}{connector}{item.name}{'/' if item.is_dir() else ''}")

        # 如果是目录，递归调用
        if item.is_dir():
            # 如果是最后一个元素，下级缩进不再需要垂直线
            new_indent = indent + ("    " if is_last else "│   ")
            draw_tree(item, new_indent)


if __name__ == "__main__":
    # --- 用户交互部分 ---
    user_input = input("请输入目标文件夹路径 (直接回车表示当前目录): ").strip()

    # 如果用户没输入，默认当前目录 '.'
    target_path = user_input if user_input else "."
    path_obj = pathlib.Path(target_path)

    # 路径检查
    if path_obj.exists() and path_obj.is_dir():
        print(f"\n{path_obj.absolute().name}/")
        draw_tree(path_obj)
    else:
        print(f"错误: 路径 '{target_path}' 不存在或不是一个文件夹。")