#!/usr/bin/env python3
"""
部署前文件检查脚本
检查所有必要的文件是否存在，用于GitHub上传前验证
"""

import os
import sys
from pathlib import Path

def check_file_exists(file_path, description):
    """检查文件是否存在"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        size_str = f"{size/1024/1024:.1f}MB" if size > 1024*1024 else f"{size/1024:.1f}KB"
        print(f"✅ {description}: {file_path} ({size_str})")
        return True
    else:
        print(f"❌ {description}: {file_path} - 文件不存在！")
        return False

def check_directory_exists(dir_path, description):
    """检查目录是否存在"""
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
        print(f"✅ {description}: {dir_path} ({file_count} 个文件)")
        return True
    else:
        print(f"❌ {description}: {dir_path} - 目录不存在！")
        return False

def main():
    print("🔍 MIDI装饰音生成器 - 部署前文件检查")
    print("=" * 50)
    
    # 检查当前目录
    current_dir = os.getcwd()
    print(f"📁 当前目录: {current_dir}")
    
    if not current_dir.endswith('pertok-ornament-generation-'):
        print("⚠️  警告：请确保在项目根目录运行此脚本")
    
    print("\n📋 检查核心Python文件...")
    core_files = [
        ("requirements.txt", "Python依赖包列表"),
        ("inference.py", "推理引擎"),
        ("ornament_model.py", "模型定义"),
        ("working_pertok_config.py", "配置文件"),
        ("fixed_pertok_decoder.py", "解码器"),
        ("ornament_aware_loss.py", "损失函数"),
        (".gitignore", "Git忽略文件"),
    ]
    
    missing_core = 0
    for file_path, description in core_files:
        if not check_file_exists(file_path, description):
            missing_core += 1
    
    print("\n📋 检查Web应用文件...")
    web_files = [
        ("web/app.py", "Flask应用主文件"),
        ("web/start.py", "生产环境启动脚本"),
        ("web/Procfile", "Render部署配置"),
        ("web/templates/index.html", "主页面模板"),
    ]
    
    missing_web = 0
    for file_path, description in web_files:
        if not check_file_exists(file_path, description):
            missing_web += 1
    
    print("\n📋 检查静态资源目录...")
    static_dirs = [
        ("web/static/css", "CSS样式文件目录"),
        ("web/static/js", "JavaScript文件目录"),
        ("web/static/uploads", "上传文件目录"),
        ("web/static/results", "结果文件目录"),
        ("web/static/scores", "乐谱文件目录"),
    ]
    
    missing_dirs = 0
    for dir_path, description in static_dirs:
        if not check_directory_exists(dir_path, description):
            missing_dirs += 1
    
    print("\n📋 检查模型文件...")
    model_file = "checkpoints_ornament_aware/best_ornament_aware_model.pth"
    model_exists = check_file_exists(model_file, "训练好的模型文件")
    
    if model_exists:
        model_size = os.path.getsize(model_file) / 1024 / 1024
        if model_size < 50:
            print(f"⚠️  警告：模型文件可能太小 ({model_size:.1f}MB)")
        elif model_size > 500:
            print(f"⚠️  警告：模型文件可能太大 ({model_size:.1f}MB)")
    
    print("\n📋 检查示例文件...")
    demo_files = [
        ("demo_input.mid", "示例输入MIDI"),
        ("demo_output.mid", "示例输出MIDI"),
    ]
    
    for file_path, description in demo_files:
        check_file_exists(file_path, description)
    
    print("\n📋 检查部署文档...")
    doc_files = [
        ("DEPLOYMENT_GUIDE.md", "详细部署指南"),
        ("DEPLOYMENT_CHECKLIST.md", "部署检查清单"),
        ("README_DEPLOYMENT.md", "项目展示页面"),
        ("GIT_COMMANDS_REFERENCE.md", "Git命令参考"),
    ]
    
    for file_path, description in doc_files:
        check_file_exists(file_path, description)
    
    # 总结
    print("\n" + "=" * 50)
    print("📊 检查结果总结:")
    
    total_missing = missing_core + missing_web + missing_dirs + (0 if model_exists else 1)
    
    if total_missing == 0:
        print("🎉 所有必要文件都存在！可以开始Git上传流程。")
        print("\n📝 下一步操作：")
        print("1. 执行: git add .")
        print("2. 执行: git status (检查文件列表)")
        print("3. 执行: git commit -m 'Initial commit for deployment'")
        print("4. 执行: git push -u origin main")
    else:
        print(f"❌ 发现 {total_missing} 个问题需要解决")
        print("\n🔧 建议操作：")
        if missing_core > 0:
            print("- 检查核心Python文件是否在正确位置")
        if missing_web > 0:
            print("- 检查Web应用文件是否完整")
        if missing_dirs > 0:
            print("- 创建缺失的目录")
        if not model_exists:
            print("- 确认模型文件路径正确")
    
    # 计算总大小
    total_size = 0
    for root, dirs, files in os.walk('.'):
        # 跳过.git目录和__pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.git') and d != '__pycache__']
        for file in files:
            if not file.endswith('.pyc'):
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                except:
                    pass
    
    total_size_mb = total_size / 1024 / 1024
    print(f"\n📦 项目总大小: {total_size_mb:.1f}MB")
    
    if total_size_mb > 1000:
        print("⚠️  警告：项目大小超过1GB，可能影响上传速度")
    elif total_size_mb < 100:
        print("⚠️  警告：项目大小可能太小，请检查模型文件")
    else:
        print("✅ 项目大小合理")

if __name__ == "__main__":
    main()