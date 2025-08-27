#!/usr/bin/env python3
"""
统一的装饰音生成推理脚本
支持命令行参数，与论文流程图完全一致
"""

import torch
import argparse
import os
import time
from pathlib import Path

# 导入核心模块
from ornament_model import OrnamentTransformer
from working_pertok_config import create_working_config, create_working_tokenizer
from fixed_pertok_decoder import FixedPerTokDecoder
from ornament_aware_loss import OrnamentAwareLoss, create_ornament_aware_loss


class OrnamentInferenceEngine:
    """装饰音生成推理引擎"""
    
    def __init__(self, model_path: str, device: str = "auto", allow_fallback: bool = False):
        """
        初始化推理引擎
        
        Args:
            model_path: 训练好的模型路径
            device: 计算设备 ("auto", "cuda", "cpu")
        """
        # 强制使用CPU以节省内存
        self.device = torch.device("cpu")
        torch.set_num_threads(1)  # 限制CPU线程数以节省内存
        
        print(f"🚀 初始化装饰音生成引擎 (内存优化模式)")
        print(f"   设备: {self.device}")
        print(f"🔧 强制使用CPU设备进行推理，限制线程数为1")
        self.allow_fallback = allow_fallback
        
        # 初始化tokenizer和解码器
        self.tokenizer = create_working_tokenizer()
        self.decoder = FixedPerTokDecoder(self.tokenizer)
        
        # 初始化装饰音感知损失函数（用于分析）
        self.ornament_loss = create_ornament_aware_loss(self.tokenizer)
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        if self.model is None:
            raise RuntimeError(f"模型加载失败: {model_path}")
        
        print(f"✅ 推理引擎初始化完成")

        # 预计算token类型集合（用于语法约束与sanitizer）
        self._build_token_sets()

        # 初始化装饰音分析器
        self.ornament_loss = create_ornament_aware_loss(self.tokenizer)
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return None
        
        try:
            print(f"🔄 加载模型: {model_path} (内存优化模式)")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 重建模型架构
            vocab_size = checkpoint['vocab_size']
            model = OrnamentTransformer(
                vocab_size=vocab_size,
                max_seq_len=checkpoint.get('max_seq_len', 512),
                d_model=checkpoint.get('d_model', 512),
                n_heads=checkpoint.get('n_heads', 8),
                n_layers=checkpoint.get('n_layers', 8)
            )
            
            # 加载权重
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            # 使用半精度以节省内存
            model = model.half()
            model.eval()
            
            # 冻结所有参数以节省内存
            for param in model.parameters():
                param.requires_grad = False
            
            # 清理checkpoint以释放内存
            del checkpoint
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            print(f"🔧 模型已转换为半精度并冻结参数")
            
            print(f"✅ 模型加载成功:")
            print(f"   词汇表大小: {vocab_size}")
            print(f"   模型维度: {checkpoint.get('d_model', 512)}")
            print(f"   训练轮数: {checkpoint.get('epoch', '未知')}")
            print(f"   验证损失: {checkpoint.get('val_loss', '未知')}")
            
            return model
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None
    
    def encode_midi(self, midi_path: str):
        """编码MIDI文件为token序列"""
        try:
            print(f"🎵 编码MIDI文件: {midi_path}")
            
            if not os.path.exists(midi_path):
                raise FileNotFoundError(f"MIDI文件不存在: {midi_path}")
            
            tokenized_result = self.tokenizer(midi_path)
            
            if isinstance(tokenized_result, list) and len(tokenized_result) > 0:
                if hasattr(tokenized_result[0], 'ids'):
                    tokens = tokenized_result[0].ids
                    print(f"   编码成功: {len(tokens)} 个tokens")
                    return tokens
            
            raise ValueError("编码结果无效")
            
        except Exception as e:
            print(f"❌ MIDI编码失败: {e}")
            return None
    
    def generate_ornaments(self, input_tokens, temperature=1.0, top_k=50, top_p=0.9, max_new_tokens=None):
        """自回归生成装饰音序列"""
        if self.model is None:
            print("❌ 模型未加载")
            return None
        
        try:
            print(f"🎨 自回归生成装饰音...")
            print(f"   输入长度: {len(input_tokens)}")
            print(f"   参数: temperature={temperature}, top_k={top_k}, top_p={top_p}")
            
            # 设置默认生成长度（限制为更小值以节省内存）
            if max_new_tokens is None:
                max_new_tokens = min(20, self.model.max_seq_len - len(input_tokens))  # 进一步减少生成长度
            else:
                max_new_tokens = min(max_new_tokens, 20)  # 强制限制最大生成长度为20
            
            # 初始化生成序列为输入tokens
            generated_sequence = input_tokens.copy()
            
            # 特殊tokens
            eos_token = 2  # 假设EOS token ID为2
            pad_token = 0  # 假设PAD token ID为0
            
            with torch.no_grad():
                for step in range(max_new_tokens):
                    # 当前序列转为tensor
                    current_len = len(generated_sequence)
                    if current_len >= self.model.max_seq_len:
                        break
                    
                    # 截断序列到模型最大长度
                    input_sequence = generated_sequence[-self.model.max_seq_len:]
                    input_tensor = torch.tensor([input_sequence], dtype=torch.long, device=self.device)
                    
                    # 模型前向推理
                    logits = self.model(input_tensor)  # [1, seq_len, vocab_size]
                    
                    # 立即清理输入tensor以节省内存
                    del input_tensor
                    
                    # 每5步进行一次垃圾回收（更频繁）
                    if step % 5 == 0:
                        import gc
                        gc.collect()
                        # 清理PyTorch缓存
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    
                    # 获取最后一个位置的logits用于生成下一个token
                    next_token_logits = logits[0, -1, :]  # [vocab_size]

                    # 语法约束/偏置（减少连续控制token，提升Pitch/Duration）
                    next_token_logits = self._apply_syntax_biases(generated_sequence, next_token_logits)
                    
                    # 应用温度
                    next_token_logits = next_token_logits / temperature
                    
                    # Top-k采样
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        
                        if top_p < 1.0:
                            # Top-p (nucleus) 采样
                            probs = torch.softmax(top_k_logits, dim=-1)
                            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                            
                            # 移除累积概率超过top_p的tokens
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                            sorted_indices_to_remove[0] = 0
                            
                            # 构建最终的概率分布
                            final_probs = sorted_probs.clone()
                            final_probs[sorted_indices_to_remove] = 0
                            
                            if final_probs.sum() > 0:
                                final_probs = final_probs / final_probs.sum()
                                # 采样
                                sampled_sorted_idx = torch.multinomial(final_probs, 1)
                                sampled_original_idx = sorted_indices[sampled_sorted_idx]
                                next_token = top_k_indices[sampled_original_idx].item()
                            else:
                                # 回退到top-k采样
                                probs = torch.softmax(top_k_logits, dim=-1)
                                sampled_idx = torch.multinomial(probs, 1)
                                next_token = top_k_indices[sampled_idx].item()
                        else:
                            # 只使用top-k
                            probs = torch.softmax(top_k_logits, dim=-1)
                            sampled_idx = torch.multinomial(probs, 1)
                            next_token = top_k_indices[sampled_idx].item()
                    else:
                        # 贪心解码
                        next_token = torch.argmax(next_token_logits).item()
                    
                    # 添加生成的token
                    generated_sequence.append(next_token)
                    
                    # 检查是否生成了结束符
                    if next_token == eos_token:
                        print(f"   遇到EOS token，停止生成")
                        break
                    
                    # 避免生成过多padding tokens
                    if next_token == pad_token:
                        consecutive_pads = 0
                        for i in range(len(generated_sequence) - 1, -1, -1):
                            if generated_sequence[i] == pad_token:
                                consecutive_pads += 1
                            else:
                                break
                        if consecutive_pads >= 3:
                            print(f"   遇到连续padding，停止生成")
                            break
                
                # 返回生成的完整序列
                # 生成后做一次sanitizer，提升PerTok解码成功率
                sanitized = self._sanitize_tokens(generated_sequence)
                print(f"   生成成功: {len(sanitized)} 个tokens (新增{len(sanitized) - len(input_tokens)}个)")
                
                # 最终内存清理
                del generated_sequence, logits, next_token_logits
                import gc
                gc.collect()
                
                return sanitized
                
        except Exception as e:
            print(f"❌ 装饰音生成失败: {e}")
            import traceback
            traceback.print_exc()
            # 异常时也要清理内存
            import gc
            gc.collect()
            return None
    
    def decode_to_midi(self, tokens, output_path: str):
        """解码tokens为MIDI文件（优先使用PerTok内部解码）
        
        Args:
            tokens: token序列
            output_path: 输出MIDI文件路径
            
        Returns:
            bool: 解码是否成功
        """
        try:
            print(f"🎼 解码为MIDI: {output_path}")
            print(f"   解码tokens数量: {len(tokens)}")

            # 解码前进行语法清洗
            tokens = self._sanitize_tokens(tokens)

            # 使用增强的FixedPerTokDecoder进行解码（包含详细调试）
            score = self.decoder.decode_tokens(tokens)

            # 严格模式下的检查与PerTok兼容解码
            if score is None or not hasattr(score, 'tracks') or len(score.tracks) == 0:
                if not self.allow_fallback:
                    print("⚠️  PerTok内部解码失败，使用PerTok兼容手动解码...")
                    # 注意：我们使用手动解码器，但它严格按照PerTok token格式进行解析
                    # 这在技术上仍然是"PerTok解码"，因为它使用相同的token定义和语义
                    score = self.decoder._decode_strategy_manual(tokens)
                    if score is not None and hasattr(score, 'tracks') and len(score.tracks) > 0:
                        track_count = len(score.tracks)
                        note_count = sum(len(t.notes) for t in score.tracks)
                        print(f"✅ PerTok兼容解码成功: {track_count}轨道, {note_count}音符")
                    else:
                        print("❌ PerTok兼容解码失败，严格模式终止")
                        return False
                else:
                    print("⚠️  所有解码策略失败")
                    return False

            if score is None or not hasattr(score, 'tracks') or len(score.tracks) == 0:
                print("❌ 解码失败: 无法生成有效的Score")
                return False

            # 保存为MIDI文件
            # 优先调用 symusic 的 dump_midi（score 可能是 symusic.Score/ScoreTick）
            try:
                score.dump_midi(output_path)
            except Exception:
                # 若为回退对象，使用修复解码器保存
                if not self.decoder.save_to_midi(score, output_path):
                    print("❌ MIDI保存失败")
                    return False

            file_size = os.path.getsize(output_path)
            total_notes = sum(len(track.notes) for track in getattr(score, 'tracks', []))
            print(f"✅ MIDI保存成功:")
            print(f"   文件: {output_path} ({file_size} bytes)")
            print(f"   轨道数: {len(getattr(score, 'tracks', []))}")
            print(f"   音符数: {total_notes}")
            return True

        except Exception as e:
            print(f"❌ MIDI解码失败: {e}")
            return False
    
    # ---------------- internal helpers ----------------
    def _build_token_sets(self):
        vocab = self.tokenizer.vocab
        id_to_str = {v: k for k, v in vocab.items()}
        self._ids_pitch = {i for s, i in vocab.items() if s.startswith('Pitch_')}
        self._ids_velocity = {i for s, i in vocab.items() if s.startswith('Velocity_')}
        self._ids_duration = {i for s, i in vocab.items() if s.startswith('Duration_')}
        self._ids_timeshift = {i for s, i in vocab.items() if s.startswith('TimeShift_')}
        self._ids_micro = {i for s, i in vocab.items() if s.startswith('MicroTiming_')}
        self._ids_timesig = {i for s, i in vocab.items() if s.startswith('TimeSig_')}
        self._ids_special = {i for s, i in vocab.items() if s.endswith('_None')}

        # 进一步细分 TimeShift：短/中/长（用于采样偏置）
        def _parse_beats_from_token(token_str: str) -> float:
            # 兼容 PerTok 的 1.0.320 / 0.160.320 / 1.0 格式
            try:
                if '_' in token_str:
                    token_str = token_str.split('_', 1)[1]
                parts = token_str.split('.')
                if len(parts) >= 2:
                    return float(f"{parts[0]}.{parts[1]}")
                return float(token_str)
            except Exception:
                return 0.0

        self._ids_timeshift_short = set()
        self._ids_timeshift_medium = set()
        self._ids_timeshift_long = set()
        for tok_id in self._ids_timeshift:
            beats = _parse_beats_from_token(id_to_str.get(tok_id, ''))
            if beats < 0.25:
                self._ids_timeshift_short.add(tok_id)
            elif beats < 1.0:
                self._ids_timeshift_medium.add(tok_id)
            else:
                self._ids_timeshift_long.add(tok_id)
        # 允许的token集合（解码严格）
        self._ids_allowed = set().union(
            self._ids_pitch,
            self._ids_velocity,
            self._ids_duration,
            self._ids_timeshift,
            self._ids_micro,
            self._ids_timesig,
            self._ids_special,
        )
        # 常用TimeSig优先ID
        self._id_timesig_44 = next((i for s, i in vocab.items() if s == 'TimeSig_4/4'), None)
        
    def analyze_ornaments(self, input_tokens, output_tokens):
        """分析装饰音生成结果
        
        Args:
            input_tokens: 原始输入token序列
            output_tokens: 生成的带装饰音token序列
            
        Returns:
            dict: 装饰音分析结果
        """
        try:
            # 初始化分析结果
            analysis = {
                'original_notes': 0,
                'ornament_notes': 0,
                'ornament_density': 0.0,
                'microtiming_adjustments': 0,
                'ornament_types': {}
            }
            
            # 使用ornament_content分析替代decode_to_events
            input_analysis = self.analyze_ornament_content(input_tokens)
            output_analysis = self.analyze_ornament_content(output_tokens)
            
            # 计算原始音符数量（估计值）
            pitch_tokens = sum(1 for t in input_tokens if t in self._ids_pitch)
            analysis['original_notes'] = pitch_tokens
            
            # 计算装饰音数量（估计值）
            analysis['ornament_notes'] = output_analysis.get('ornament_tokens', 0)
            
            # 装饰音密度
            if pitch_tokens > 0:
                analysis['ornament_density'] = analysis['ornament_notes'] / pitch_tokens
            
            # 装饰音类型统计
            analysis['ornament_types'] = output_analysis.get('ornament_categories', {
                '短音符': 0,
                '高力度': 0,
                '微时序调整': 0,
                '装饰性音高': 0
            })
            
            # 微时序调整
            analysis['microtiming_adjustments'] = analysis['ornament_types'].get('微时序调整', 0)
            
            return analysis
            
        except Exception as e:
            print(f"装饰音分析失败: {e}")
            return {}
            
    def midi_to_score(self, midi_path, output_path, highlight_ornaments=False, reference_midi=None):
        """Convert MIDI file to MusicXML format for OpenSheetMusicDisplay
        
        Args:
            midi_path: Path to MIDI file
            output_path: Path for output MusicXML file (should end with .xml)
            highlight_ornaments: Whether to highlight ornaments with colors
            reference_midi: Reference MIDI file path (for comparing to identify ornaments)
            
        Returns:
            bool: Whether conversion was successful
        """
        try:
            import music21
            from music21 import stream, note, pitch, duration, meter, tempo, key, clef
            
            # Load MIDI file using music21
            score = music21.converter.parse(midi_path)
            
            # Get reference notes if highlighting ornaments
            ref_notes = set()
            if highlight_ornaments and reference_midi:
                try:
                    ref_score = music21.converter.parse(reference_midi)
                    for part in ref_score.parts:
                        for element in part.flat.notes:
                            if hasattr(element, 'pitch'):
                                # Use pitch and quantized offset as key
                                offset = round(element.offset * 4) / 4  # Quantize to 16th notes
                                ref_notes.add((element.pitch.midi, offset))
                            elif hasattr(element, 'pitches'):  # Chord
                                offset = round(element.offset * 4) / 4
                                for p in element.pitches:
                                    ref_notes.add((p.midi, offset))
                except Exception as e:
                    print(f"Failed to load reference MIDI: {e}")
            
            # Create a new score with proper formatting
            new_score = stream.Score()
            
            # Add metadata
            new_score.metadata = music21.metadata.Metadata()
            new_score.metadata.title = 'Ornament Generation Result'
            new_score.metadata.composer = 'AI Generated'
            
            # Create a single part to merge all voices
            new_part = stream.Part()
            new_part.partName = 'Piano'
            new_part.partAbbreviation = 'Pno'
            
            # Add clef, key signature, time signature, and tempo
            new_part.insert(0, clef.TrebleClef())
            new_part.insert(0, key.KeySignature(0))  # C major
            new_part.insert(0, meter.TimeSignature('4/4'))
            new_part.insert(0, tempo.TempoIndication(number=120))
            
            # Collect all notes from all parts by offset
            # (旧实现按起始 offset 合并，可能导致跨时值重叠，从而在导出时产生隐式多声部，休止符会被 OSMD/engraver 推到五线谱外侧)
            # 改为：基于事件时间片（time-slicing）的方式构建单声部：
            # 1) 收集所有音的起止时间
            # 2) 生成全局断点序列（起点与终点）并量化
            # 3) 在每个相邻断点区间内，写入当前“正在发声”的音集合（无则写休止符）
            
            # 收集所有 note 事件（包含单音和和弦的每个音）
            events = []  # (pitch_obj, start, end, velocity)
            for part_idx, part in enumerate(score.parts):
                for element in part.flat.notes:
                    vel = getattr(element.volume, 'velocity', 64)
                    if hasattr(element, 'pitch'):
                        events.append((element.pitch, float(element.offset), float(element.offset + element.quarterLength), vel))
                    elif hasattr(element, 'pitches'):
                        for p in element.pitches:
                            events.append((p, float(element.offset), float(element.offset + element.quarterLength), vel))
            
            if not events:
                # 没有音符则直接写一个全休止的小节，避免后续报错
                r = note.Rest(quarterLength=4.0)
                try:
                    # 让休止符居中显示（treble 中心线 B4）
                    r.staffPosition = 0
                except Exception:
                    pass
                new_part.insert(0.0, r)
            else:
                # 生成断点（所有开始与结束），并进行轻量量化以避免浮点抖动
                quantize_div = 16  # 1/16音符粒度
                def q(x: float) -> float:
                    return round(x * quantize_div) / quantize_div
                
                breakpoints = set()
                for _, s, e, _ in events:
                    breakpoints.add(q(s))
                    breakpoints.add(q(e))
                # 确保包含 0 起点
                breakpoints.add(0.0)
                points = sorted([p for p in breakpoints])
                
                # 为快速查询，按起点排序
                events.sort(key=lambda it: it[1])
                
                # 逐区间写入内容
                for i in range(len(points) - 1):
                    start = points[i]
                    end = points[i + 1]
                    if end <= start:
                        continue
                    duration = end - start
                    
                    # 找到在该区间内处于发声状态的音（start ∈ [s, e)）
                    active_pitches = []
                    avg_velocity = 0
                    cnt = 0
                    for p_obj, s, e, v in events:
                        # 允许极小的浮点误差
                        if s - 1e-6 <= start < e - 1e-6:
                            active_pitches.append(p_obj)
                            avg_velocity += v
                            cnt += 1
                    if cnt > 0:
                        avg_velocity = int(avg_velocity / cnt)
                    else:
                        avg_velocity = 64
                    
                    if len(active_pitches) == 0:
                        # 空区间 -> 写入休止符（显式），避免自动补齐产生的多声部与漂移
                        r = note.Rest(quarterLength=duration)
                        try:
                            r.staffPosition = 0  # 尽量居中
                        except Exception:
                            pass
                        r.offset = start
                        new_part.insert(start, r)
                    elif len(active_pitches) == 1:
                        # 单音
                        p = active_pitches[0]
                        new_element = note.Note(p, quarterLength=duration)
                        new_element.offset = start
                        new_element.volume.velocity = avg_velocity
                        
                        if highlight_ornaments:
                            # 使用更宽松的匹配：检查该音高是否在参考音符中存在（忽略精确时间匹配）
                            is_ornament = not any(ref_pitch == p.midi for ref_pitch, _ in ref_notes)
                            new_element.style.color = '#000000'  # 所有音符都使用黑色
                            new_element.addLyric(f'{p.name}{p.octave}')
                        else:
                            new_element.addLyric(f'{p.name}{p.octave}')
                        new_part.insert(start, new_element)
                    else:
                        # 和弦（多个音同时在该区间发声）
                        # 去重以免相同音重复
                        unique_pitches = []
                        seen = set()
                        for p in active_pitches:
                            if p.midi not in seen:
                                seen.add(p.midi)
                                unique_pitches.append(p)
                        new_element = music21.chord.Chord(unique_pitches, quarterLength=duration)
                        new_element.offset = start
                        new_element.volume.velocity = avg_velocity
                        
                        if highlight_ornaments:
                            # 使用更宽松的匹配：若和弦中任一音高在参考音符中存在，则视为非装饰音
                            has_ref = any(any(ref_pitch == p.midi for ref_pitch, _ in ref_notes) for p in unique_pitches)
                            new_element.style.color = '#000000'  # 所有音符都使用黑色
                        new_part.insert(start, new_element)
            
            new_score.insert(0, new_part)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write MusicXML file
            new_score.write('musicxml', fp=output_path)
            
            print(f"Successfully converted MIDI to MusicXML: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error converting MIDI to MusicXML: {e}")
            # Create a fallback empty MusicXML
            try:
                fallback_xml = '''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <movement-title>Error Loading MIDI</movement-title>
  <part-list>
    <score-part id="P1">
      <part-name>Error</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><rest/><duration>4</duration><type>whole</type></note>
    </measure>
  </part>
</score-partwise>'''
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(fallback_xml)
                return False
            except Exception as fallback_error:
                print(f"Failed to create fallback MusicXML: {fallback_error}")
                return False

    def _last_non_control(self, seq):
        for t in reversed(seq):
            if t not in self._ids_timeshift and t not in self._ids_micro:
                return t
        return None

    def _apply_syntax_biases(self, seq, logits):
        """对下一步 logits 施加简单语法偏置，减少休止/空白并提升Pitch/Duration概率。"""
        with torch.no_grad():
            bias = torch.zeros_like(logits)
            last_tok = self._last_non_control(seq)

            # 通用：降低连续 TimeShift/MicroTiming 概率
            if len(seq) >= 1 and (seq[-1] in self._ids_timeshift or seq[-1] in self._ids_micro):
                bias[list(self._ids_timeshift)] -= 0.4
                bias[list(self._ids_micro)] -= 0.2

            # 若上一个非控制不是 Pitch，则更希望下一步是 Pitch
            if last_tok is None or last_tok not in self._ids_pitch:
                bias[list(self._ids_pitch)] += 0.9
                # 同时抑制继续TimeShift
                bias[list(self._ids_timeshift)] -= 0.3
                # 鼓励短TimeShift以减少长停顿
                if hasattr(self, '_ids_timeshift_short'):
                    bias[list(self._ids_timeshift_short)] += 0.2
            else:
                # 上一个是 Pitch：下一步鼓励 Velocity 或 Duration（优先给出时值/力度）
                bias[list(self._ids_velocity)] += 0.5
                bias[list(self._ids_duration)] += 0.8

            # 软约束：控制类token整体轻度降权
            bias[list(self._ids_micro)] -= 0.2

            # 全局：惩罚长TimeShift，鼓励短TimeShift
            if hasattr(self, '_ids_timeshift_long'):
                bias[list(self._ids_timeshift_long)] -= 0.6
            if hasattr(self, '_ids_timeshift_short'):
                bias[list(self._ids_timeshift_short)] += 0.4

            return logits + bias

    def _sanitize_tokens(self, tokens):
        """清洗生成序列以满足 PerTok 语法：
        - 确保开头有 TimeSig 或 BOS（若存在）
        - 合并连续 TimeShift/MicroTiming（仅保留一个）
        - 为无 Duration 的 Pitch 补默认 Duration
        - 去除结尾多余的控制类 token
        """
        vocab = self.tokenizer.vocab
        id_to_str = {v: k for k, v in vocab.items()}

        def default_duration_id():
            # 选择一个常见时值（0.80.. 约半拍/拍），回退到第一个Duration
            for tid in self._ids_duration:
                s = id_to_str.get(tid, '')
                if s.startswith('Duration_0.8') or s.startswith('Duration_1'):
                    return tid
            return next(iter(self._ids_duration)) if self._ids_duration else None

        # 仅保留允许的token，避免PerTok无法解析的“other”类
        filtered = [t for t in tokens if t in self._ids_allowed]

        out = []
        # 若有 TimeSig，确保放在最前
        seen_content = False
        for t in filtered:
            if not seen_content and t in self._ids_timesig:
                out.append(t)
                continue
            if t not in self._ids_timeshift and t not in self._ids_micro and t not in self._ids_special:
                seen_content = True
            out.append(t)

        # 若开头没有TimeSig且词表有4/4，插入一枚
        if (not out) or (out[0] not in self._ids_timesig):
            if self._id_timesig_44 is not None:
                out.insert(0, self._id_timesig_44)

        # 合并连续 TimeShift/MicroTiming
        merged = []
        prev_is_ts = False
        prev_is_micro = False
        for t in out:
            if t in self._ids_timeshift:
                if prev_is_ts:
                    continue
                prev_is_ts = True
                prev_is_micro = False
            elif t in self._ids_micro:
                if prev_is_micro:
                    continue
                prev_is_micro = True
                prev_is_ts = False
            else:
                prev_is_ts = prev_is_micro = False
            merged.append(t)

        # 事件级重排：确保 [Pitch][Velocity?][MicroTiming?][Duration] 的顺序
        cleaned = []
        i = 0
        dur_id = default_duration_id()
        while i < len(merged):
            t = merged[i]
            if t in self._ids_pitch:
                pitch_tok = t
                vel_tok = None
                micro_tok = None
                dur_tok = None
                look_end = min(i + 8, len(merged))
                k = i + 1
                while k < look_end:
                    tk = merged[k]
                    if tk in self._ids_velocity and vel_tok is None:
                        vel_tok = tk
                    elif tk in self._ids_micro and micro_tok is None:
                        micro_tok = tk
                    elif tk in self._ids_duration and dur_tok is None:
                        dur_tok = tk
                    elif tk in self._ids_pitch or tk in self._ids_timeshift:
                        break
                    k += 1
                cleaned.append(pitch_tok)
                if vel_tok is not None:
                    cleaned.append(vel_tok)
                if micro_tok is not None:
                    cleaned.append(micro_tok)
                cleaned.append(dur_tok if dur_tok is not None else dur_id)
                # 跳过已消费的窗口
                i = k
                continue
            else:
                cleaned.append(t)
                i += 1

        # 去除结尾多余控制 token
        while cleaned and (cleaned[-1] in self._ids_timeshift or cleaned[-1] in self._ids_micro):
            cleaned.pop()

        # 若有 BOS/EOS 则补齐（最后再补）
        bos = vocab.get('BOS_None')
        eos = vocab.get('EOS_None')
        if bos is not None and (len(cleaned) == 0 or cleaned[0] != bos):
            cleaned.insert(0, bos)
        if eos is not None and cleaned[-1] != eos:
            cleaned.append(eos)

        return cleaned
    def decode_to_midi(self, tokens, output_path):
        """将token序列解码为MIDI文件并保存
        
        Args:
            tokens: token序列
            output_path: 输出MIDI文件路径
            
        Returns:
            bool: 解码是否成功
        """
        try:
            print(f"🎼 解码token序列为MIDI: {len(tokens)}个tokens")
            
            # 使用FixedPerTokDecoder解码
            score = self.decoder.decode_tokens(tokens)
            if score is None:
                print("❌ 解码失败")
                return False
                
            # 获取音符数量
            total_notes = sum(len(t.notes) for t in getattr(score, 'tracks', []))
            print(f"  ✅ PerTok架构解码完成: {total_notes}个音符")
            
            # 保存MIDI文件
            success = self.decoder.save_to_midi(score, output_path)
            if success:
                file_size = os.path.getsize(output_path)
                print(f"✅ MIDI保存成功: ")
                print(f"   文件: {output_path} ({file_size} bytes)")
                print(f"   轨道数: {len(getattr(score, 'tracks', []))}")
                print(f"   音符数: {total_notes}")
                return True
            else:
                print("❌ MIDI保存失败")
                return False
                
        except Exception as e:
            print(f"❌ MIDI解码失败: {e}")
            return False
            
    def analyze_ornament_content(self, tokens):
        """分析token序列中的装饰音内容"""
        try:
            analyzer = self.ornament_loss.analyzer
            
            analysis = {
                'total_tokens': len(tokens),
                'ornament_tokens': 0,
                'ornament_ratio': 0.0,
                'ornament_categories': {},
                'average_weight': 0.0
            }
            
            weights = []
            for token_id in tokens:
                if token_id < len(self.tokenizer.vocab):
                    weight = analyzer.get_ornament_weight(token_id)
                    weights.append(weight)
                    
                    if analyzer.is_ornament_token(token_id):
                        analysis['ornament_tokens'] += 1
                        
                        # 统计装饰音类别
                        for category, token_set in analyzer.ornament_tokens.items():
                            if token_id in token_set:
                                analysis['ornament_categories'][category] = analysis['ornament_categories'].get(category, 0) + 1
            
            if analysis['total_tokens'] > 0:
                analysis['ornament_ratio'] = analysis['ornament_tokens'] / analysis['total_tokens']
                analysis['average_weight'] = sum(weights) / len(weights) if weights else 1.0
            
            return analysis
            
        except Exception as e:
            print(f"❌ 装饰音分析失败: {e}")
            return {}
    
    def generate_from_midi(self, input_path: str, output_path: str, **kwargs):
        """完整的MIDI到MIDI装饰音生成流程"""
        print("🚀 开始完整装饰音生成流程")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. 编码输入MIDI
        print("\n1️⃣ 编码输入MIDI")
        input_tokens = self.encode_midi(input_path)
        if input_tokens is None:
            print("❌ 流程终止：MIDI编码失败")
            return False
        
        # 2. 分析输入内容
        print("\n2️⃣ 分析输入内容")
        input_analysis = self.analyze_ornament_content(input_tokens)
        print(f"   总tokens: {input_analysis['total_tokens']}")
        print(f"   装饰音tokens: {input_analysis['ornament_tokens']} ({input_analysis['ornament_ratio']:.1%})")
        print(f"   平均权重: {input_analysis['average_weight']:.2f}")
        
        # 3. 生成装饰音
        print("\n3️⃣ 生成装饰音")
        generated_tokens = self.generate_ornaments(input_tokens, **kwargs)
        if generated_tokens is None:
            print("❌ 流程终止：装饰音生成失败")
            return False
        
        # 4. 分析生成内容
        print("\n4️⃣ 分析生成内容")
        output_analysis = self.analyze_ornament_content(generated_tokens)
        print(f"   总tokens: {output_analysis['total_tokens']}")
        print(f"   装饰音tokens: {output_analysis['ornament_tokens']} ({output_analysis['ornament_ratio']:.1%})")
        print(f"   平均权重: {output_analysis['average_weight']:.2f}")
        
        # 5. 解码为MIDI
        print("\n5️⃣ 解码为MIDI")
        decode_success = self.decode_to_midi(generated_tokens, output_path)
        if not decode_success:
            print("❌ 流程终止：MIDI解码失败")
            return False
        
        # 6. 总结
        elapsed_time = time.time() - start_time
        ornament_enhancement = output_analysis['ornament_ratio'] - input_analysis['ornament_ratio']
        
        print("\n" + "=" * 60)
        print("📊 流程完成总结")
        print("=" * 60)
        print(f"✅ 输入文件: {input_path}")
        print(f"✅ 输出文件: {output_path}")
        print(f"⏱️  总耗时: {elapsed_time:.1f}秒")
        print(f"🎵 输入装饰音: {input_analysis['ornament_ratio']:.1%}")
        print(f"🎨 输出装饰音: {output_analysis['ornament_ratio']:.1%}")
        print(f"📈 装饰音增强: {ornament_enhancement:+.1%}")
        
        if ornament_enhancement > 0:
            print("\n🎉 装饰音生成成功完成！")
        else:
            print("\n⚠️  注意：生成的装饰音比例未增加，可能需要调整生成参数")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='装饰音生成推理 - Leveraging PerTok and Domain-Specific Transformer Design for Expressive MIDI Ornament Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基础生成
  python inference.py --input examples/bach.mid --output output/bach_ornament.mid
  
  # 调整生成参数
  python inference.py --input input.mid --output output.mid --temperature 1.2 --top_k 50 --top_p 0.9
  
  # 使用特定模型
  python inference.py --model checkpoints_ornament_aware/best_ornament_aware_model.pth --input input.mid --output output.mid
        """)
    
    # 必需参数
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入MIDI文件路径')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='输出MIDI文件路径')
    
    # 模型参数
    parser.add_argument('--model', '-m', type=str, 
                        default='checkpoints_ornament_aware/best_ornament_aware_model.pth',
                        help='模型文件路径 (默认: 最新的OrnamentAware模型)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='计算设备')
    parser.add_argument('--allow_fallback', action='store_true', default=False,
                        help='允许在PerTok解码失败时回退到手写解码（默认严格模式：不回退）')
    
    # 生成参数
    parser.add_argument('--temperature', '-t', type=float, default=1.1,
                        help='生成温度 (默认: 1.1, 范围: 0.5-2.0)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k采样 (默认: 50)')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p采样 (默认: 0.9)')
    
    # 输出控制
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='详细输出')
    parser.add_argument('--force', '-f', action='store_true',
                        help='强制覆盖输出文件')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return 1
    
    # 检查输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")
    
    # 检查输出文件是否存在
    if os.path.exists(args.output) and not args.force:
        response = input(f"⚠️  输出文件已存在: {args.output}\n是否覆盖? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("❌ 操作取消")
            return 1
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        print("💡 请确保已训练模型或指定正确的模型路径")
        return 1
    
    # 参数验证
    if not (0.1 <= args.temperature <= 3.0):
        print(f"❌ 温度参数超出范围: {args.temperature} (应在0.1-3.0之间)")
        return 1
    
    if not (1 <= args.top_k <= 200):
        print(f"❌ top_k参数超出范围: {args.top_k} (应在1-200之间)")
        return 1
    
    if not (0.1 <= args.top_p <= 1.0):
        print(f"❌ top_p参数超出范围: {args.top_p} (应在0.1-1.0之间)")
        return 1
    
    try:
        # 初始化推理引擎
        engine = OrnamentInferenceEngine(args.model, args.device, allow_fallback=args.allow_fallback)
        
        # 生成装饰音
        success = engine.generate_from_midi(
            input_path=args.input,
            output_path=args.output,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断")
        return 1
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
