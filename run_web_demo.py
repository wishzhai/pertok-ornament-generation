#!/usr/bin/env python3
"""
装饰音生成Web演示启动脚本

使用方法：
    python run_web_demo.py [--port 端口号] [--host 主机地址] [--model 模型路径]
"""

import os
import sys
import argparse
from pathlib import Path

# 解析命令行参数
parser = argparse.ArgumentParser(description='启动装饰音生成Web演示')
parser.add_argument('--port', type=int, default=5000, help='Web服务端口号')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Web服务主机地址')
parser.add_argument('--model', type=str, default=None, help='模型路径（默认使用最佳模型）')
parser.add_argument('--debug', action='store_true', help='启用调试模式')
args = parser.parse_args()

# 设置模型路径
if args.model is None:
    args.model = os.path.join('checkpoints_ornament_aware', 'best_ornament_aware_model.pth')

# 检查模型文件是否存在
if not os.path.exists(args.model):
    print(f"❌ 错误: 模型文件不存在: {args.model}")
    sys.exit(1)

# 检查依赖项
try:
    import flask
    import music21
    import torch
    from inference import OrnamentInferenceEngine
except ImportError as e:
    print(f"❌ 错误: 缺少依赖项: {e}")
    print("请先安装所需依赖: pip install -r requirements.txt")
    sys.exit(1)

# 设置环境变量
os.environ['ORNAMENT_MODEL_PATH'] = os.path.abspath(args.model)

print(f"🎵 装饰音生成Web演示")
print(f"   模型路径: {os.environ['ORNAMENT_MODEL_PATH']}")
print(f"   主机地址: {args.host}")
print(f"   端口号: {args.port}")
print(f"   调试模式: {'启用' if args.debug else '禁用'}")

# 启动Web服务
from web.app import app
app.run(host=args.host, port=args.port, debug=args.debug)