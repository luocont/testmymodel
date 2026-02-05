# -*- coding: utf-8 -*-
"""
运行 CTRS 评估的包装脚本
"""
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CTRS'))

# 导入并运行主脚本
if __name__ == "__main__":
    import argparse

    # 解析参数
    parser = argparse.ArgumentParser(description="Evaluate CTRS results")
    parser.add_argument("-i", "--input_dir", type=str, default="output-cn",
                        help="Directory to read the sessions")
    parser.add_argument("-o", "--output_dir", type=str, default="output-ctrs",
                        help="Directory to save the results")
    parser.add_argument("-m_iter", "--max_iter", type=int, default=1,
                        help="Number of times GPT-4o is run for scoring a single session")

    args = parser.parse_args()

    # 切换到 CTRS 目录
    os.chdir(os.path.join(os.path.dirname(__file__), 'CTRS'))

    # 导入主脚本
    import importlib.util
    spec = importlib.util.spec_from_file_location("ctrs_main", "ctrs-gpt4o.py")
    ctrs_module = importlib.util.module_from_spec(spec)
    sys.argv = ['ctrs-gpt4o.py', '-i', args.input_dir, '-o', args.output_dir, '-m_iter', str(args.max_iter)]
    spec.loader.exec_module(ctrs_module)
