"""仓库目录约定: 代码在 src/, 输入数据在 data/, 生成的结果在 results/。"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
