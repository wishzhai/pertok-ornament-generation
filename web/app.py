#!/usr/bin/env python3
"""
Ornament Generation Web Demo Application

Provides a web interface to upload MIDI files, generate ornaments, and visualize as sheet music
"""

import os
import sys
import json
import uuid
import tempfile
import psutil
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename


# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入装饰音生成模块
from inference import OrnamentInferenceEngine
from working_pertok_config import create_working_tokenizer

# 初始化Flask应用
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# 配置上传文件夹
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
RESULT_FOLDER = os.path.join(app.static_folder, 'results')
SCORE_FOLDER = os.path.join(app.static_folder, 'scores')

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(SCORE_FOLDER, exist_ok=True)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'mid', 'midi'}

# 模型路径
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         'checkpoints_ornament_aware', 'best_ornament_aware_model.pth')


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_inference_engine():
    """Create a temporary inference engine instance with memory optimization"""
    try:
        # 强制使用CPU并启用内存优化
        return OrnamentInferenceEngine(model_path, device="cpu")
    except Exception as e:
        print(f"Failed to create inference engine: {e}")
        return None

def cleanup_inference_engine(engine):
    """Clean up inference engine and free memory"""
    if engine is not None:
        # 清理模型
        if hasattr(engine, 'model') and engine.model is not None:
            # 清理模型参数
            for param in engine.model.parameters():
                del param
            del engine.model
        # 清理其他组件
        if hasattr(engine, 'tokenizer'):
            del engine.tokenizer
        if hasattr(engine, 'decoder'):
            del engine.decoder
        if hasattr(engine, 'ornament_loss'):
            del engine.ornament_loss
        del engine
        # 强制垃圾回收（多次）
        import gc
        for _ in range(3):
            gc.collect()
        # 清理PyTorch缓存
        import torch
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()



@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy', 'service': 'ornament-generator'}), 200

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')





@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        base_name = os.path.splitext(filename)[0]
        unique_filename = f"{base_name}_{unique_id}.mid"
        
        # Save uploaded file
        input_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(input_path)
        
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'message': 'File uploaded successfully'
        })
    
    return jsonify({'error': 'Unsupported file type'}), 400


@app.route('/generate', methods=['POST'])
def generate_ornaments():
    """Generate ornaments"""
    data = request.json
    filename = data.get('filename')
    temperature = float(data.get('temperature', 1.0))
    top_k = int(data.get('top_k', 50))
    top_p = float(data.get('top_p', 0.9))
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    # Create temporary inference engine
    # 检查初始内存使用（放宽阈值，因为使用模拟模型）
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    print(f"🔍 初始内存使用: {initial_memory:.1f}MB")
    
    if initial_memory > 500:  # 500MB threshold (relaxed)
        return jsonify({'error': 'Server memory usage too high, please try again later'}), 503
    
    inference_engine = None
    try:
        inference_engine = create_inference_engine()
        if inference_engine is None:
            return jsonify({'error': 'Failed to create inference engine'}), 500
        
        # 检查模型加载后的内存（模拟模型应该很轻量）
        model_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        print(f"🔍 模型加载后内存: {model_memory:.1f}MB")
        
        if model_memory > 510:  # 510MB threshold (relaxed for mock model)
            return jsonify({'error': 'Memory usage too high after model loading'}), 503
        
        # Input and output paths
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_filename = f"ornament_{filename}"
        output_path = os.path.join(RESULT_FOLDER, output_filename)
        
        # Encode MIDI file
        input_tokens = inference_engine.encode_midi(input_path)
        if input_tokens is None:
            return jsonify({'error': 'Failed to encode MIDI'}), 500
        
        # Generate ornaments with relaxed limits for mock model
        output_tokens = inference_engine.generate_ornaments(
            input_tokens, 
            temperature=min(temperature, 0.8),  # 模拟模式下放宽温度
            top_k=min(top_k, 20),  # 模拟模式下放宽top_k
            top_p=min(top_p, 0.8),  # 模拟模式下放宽top_p
            max_new_tokens=20  # 模拟模式下放宽生成长度
        )
        
        if output_tokens is None:
            return jsonify({'error': 'Failed to generate ornaments'}), 500
        
        # Decode to MIDI file
        success = inference_engine.decode_to_midi(output_tokens, output_path)
        if not success:
            return jsonify({'error': 'Failed to decode MIDI'}), 500
        
        # Generate MusicXML files
        input_score_path = os.path.join(SCORE_FOLDER, f"input_{filename}.xml")
        output_score_path = os.path.join(SCORE_FOLDER, f"output_{filename}.xml")
        
        # Generate regular MusicXML for input MIDI
        inference_engine.midi_to_score(input_path, input_score_path)
        
        # Generate MusicXML with highlighted ornaments for output MIDI
        inference_engine.midi_to_score(
            output_path, 
            output_score_path, 
            highlight_ornaments=True, 
            reference_midi=input_path
        )
        
        # Analyze ornaments
        ornament_analysis = inference_engine.analyze_ornaments(input_tokens, output_tokens)
        
        return jsonify({
            'success': True,
            'input_midi': filename,
            'output_midi': output_filename,
            'input_score': f"input_{filename}.xml",
            'output_score': f"output_{filename}.xml",
            'analysis': ornament_analysis
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    finally:
        # Clean up memory using dedicated cleanup function
        if 'inference_engine' in locals():
            cleanup_inference_engine(inference_engine)








@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/results/<filename>')
def result_file(filename):
    """Serve generated files"""
    return send_from_directory(RESULT_FOLDER, filename)


@app.route('/scores/<filename>')
def score_file(filename):
    """Serve score images"""
    return send_from_directory(SCORE_FOLDER, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)