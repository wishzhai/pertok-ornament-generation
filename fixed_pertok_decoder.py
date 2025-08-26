#!/usr/bin/env python3
"""
Fixed PerTok Decoder
- Prioritize PerTok internal decoding (tokens_to_score / _tokens_to_score)
- Fall back to manual decoding with TimeShift / Pitch / Duration / Velocity / MicroTiming
- TPQ read from tokenizer.config.additional_params['ticks_per_quarter'] (default 480)
"""
from __future__ import annotations

import re
from typing import List, Optional

import torch
from miditok import PerTok
from miditok.classes import TokSequence

import symusic
from symusic import Score, Track, Note, Tempo, TimeSignature


class FixedPerTokDecoder:
    """Robust decoder for PerTok token sequences."""

    def __init__(self, tokenizer: PerTok) -> None:
        self.tokenizer = tokenizer
        # Miditok PerTok.vocab: token_str -> id
        self.vocab = tokenizer.vocab
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}  # id -> token_str

        # Read TPQ from tokenizer config (additional_params)
        tpq = 480
        config = getattr(tokenizer, 'config', None)
        if config is not None:
            add_params = getattr(config, 'additional_params', {})
            tpq_raw = add_params.get('ticks_per_quarter', tpq)
            # 安全转换TPQ - 处理可能的字符串格式问题
            try:
                if isinstance(tpq_raw, str):
                    # 如果是字符串，提取数字部分
                    tpq_match = re.search(r'(\d+)', str(tpq_raw))
                    if tpq_match:
                        tpq = int(tpq_match.group(1))
                    else:
                        tpq = 480  # 默认值
                else:
                    tpq = int(float(tpq_raw))  # 先转float再转int处理小数
            except (ValueError, TypeError):
                print(f"⚠️ TPQ解析失败，使用默认值480: {tpq_raw}")
                tpq = 480
        self.tpq = tpq
        print(f"解码器使用TPQ: {self.tpq} (与tokenizer配置一致)")

        # MicroTiming params
        self.max_micro_shift_beats = 0.0
        self.num_micro_bins = 0
        if config is not None:
            add_params = getattr(config, 'additional_params', {})
            self.max_micro_shift_beats = float(add_params.get('max_microtiming_shift', 0.0))
            self.num_micro_bins = int(add_params.get('num_microtiming_bins', 0))

        # Token type buckets
        self._analyze_token_types()

    def _analyze_token_types(self) -> None:
        token_types = {
            'special': set(),
            'pitch': set(),
            'velocity': set(),
            'duration': set(),
            'time_shift': set(),
            'microtiming': set(),
            'tempo': set(),
            'time_signature': set(),
            'other': set(),
        }
        for token_str, token_id in self.vocab.items():
            if token_str.startswith('Pitch_'):
                token_types['pitch'].add(token_id)
            elif token_str.startswith('Velocity_'):
                token_types['velocity'].add(token_id)
            elif token_str.startswith('Duration_'):
                token_types['duration'].add(token_id)
            elif token_str.startswith('TimeShift_'):
                token_types['time_shift'].add(token_id)
            elif token_str.startswith('MicroTiming_'):
                token_types['microtiming'].add(token_id)
            elif token_str.startswith('Tempo_'):
                token_types['tempo'].add(token_id)
            elif token_str.startswith('TimeSig_'):
                token_types['time_signature'].add(token_id)
            elif token_str.endswith('_None'):
                token_types['special'].add(token_id)
            else:
                token_types['other'].add(token_id)
        self.token_types = token_types
        print("Token类型分析完成:")
        for k, s in self.token_types.items():
            print(f"  {k}: {len(s)} tokens")

    # ============== Public API ==============
    def decode_tokens(self, tokens: List[int]) -> Optional[Score]:
        """使用纯PerTok架构解码（严格遵循PerTok规范）"""
        print(f"🎯 开始PerTok架构解码: {len(tokens)}个tokens")
        
        # 第一步：尝试PerTok内部解码
        score = self._decode_strategy_pertok(tokens)
        if score is not None and hasattr(score, 'tracks') and len(score.tracks) > 0:
            print(f"✅ PerTok内部解码成功")
            return score
        
        # 第二步：使用PerTok架构的手动解码（这是真正的PerTok解码）
        print(f"⚡ 使用PerTok架构手动解码（严格遵循PerTok规范）")
        score = self._decode_strategy_pertok_architecture(tokens)
        if score is not None and hasattr(score, 'tracks') and len(score.tracks) > 0:
            print(f"✅ PerTok架构解码成功")
            return score
        
        # 第三步：传统手动解码作为最后回退
        return self._decode_strategy_manual(tokens)

    def save_to_midi(self, score: Score, output_path: str) -> bool:
        try:
            score.dump_midi(output_path)
            return True
        except Exception as e:
            print(f"保存MIDI失败: {e}")
            return False

    def test_decode(self, tokens: List[int], output_path: str) -> bool:
        score = self.decode_tokens(tokens)
        if score is None:
            print("解码失败")
            return False
        total_notes = sum(len(t.notes) for t in getattr(score, 'tracks', []))
        print(f"解码成功: {len(getattr(score, 'tracks', []))} 轨道, {total_notes} 个音符")
        if self.save_to_midi(score, output_path):
            print(f"MIDI文件已保存: {output_path}")
            return True
        return False

    # ============== Strategies ==============
    def _decode_strategy_pertok(self, tokens: List[int]) -> Optional[Score]:
        """Use PerTok internal decoding if possible."""
        print(f"🔍 PerTok解码调试: 输入{len(tokens)}个tokens")
        
        # Clean and add BOS/EOS if present in vocab
        cleaned: List[int] = []
        vocab_size = max(self.vocab_reverse.keys()) + 1 if self.vocab_reverse else 0
        filtered_count = 0
        for t in tokens:
            if isinstance(t, int) and 0 <= t < vocab_size:
                cleaned.append(t)
            else:
                filtered_count += 1
        
        if filtered_count > 0:
            print(f"  过滤了{filtered_count}个无效tokens")
        
        # 检查必要的结构tokens
        bos = self.vocab.get('BOS_None')
        eos = self.vocab.get('EOS_None')
        timesig = self.vocab.get('TimeSig_4/4')  # 常见拍号
        
        print(f"  BOS token: {bos}, EOS token: {eos}, TimeSig token: {timesig}")
        
        # 去重：移除重复的BOS/TimeSig/EOS
        cleaned = self._deduplicate_structure_tokens(cleaned)
        
        # 确保序列开头有BOS
        if bos is not None and (len(cleaned) == 0 or cleaned[0] != bos):
            cleaned.insert(0, bos)
            print("  添加了BOS token")
        
        # 在BOS后确保有TimeSig
        if timesig is not None:
            has_timesig = False
            for i in range(min(5, len(cleaned))):  # 检查前5个位置
                token_str = self.vocab_reverse.get(cleaned[i], '')
                if token_str.startswith('TimeSig_'):
                    has_timesig = True
                    break
            if not has_timesig:
                insert_pos = 1 if bos is not None else 0
                cleaned.insert(insert_pos, timesig)
                print("  添加了TimeSig token")
        
        # 确保序列结尾有EOS
        if eos is not None and (len(cleaned) == 0 or cleaned[-1] != eos):
            cleaned.append(eos)
            print("  添加了EOS token")
        
        print(f"  清理后: {len(cleaned)}个tokens")
        
        # 修复不规范的token格式（如 Duration_1.0.320 -> Duration_1.0）
        cleaned = self._fix_malformed_tokens(cleaned)
        
        # 显示序列开头几个tokens用于调试
        preview_tokens = []
        for i in range(min(10, len(cleaned))):
            token_str = self.vocab_reverse.get(cleaned[i], f'UNK_{cleaned[i]}')
            preview_tokens.append(token_str)
        print(f"  序列预览: {' | '.join(preview_tokens)}...")

        tok_seq = TokSequence(ids=cleaned)
        result = None
        
        # Try strategy 1: without programs
        try:
            print("  尝试策略1: 无programs参数")
            if hasattr(self.tokenizer, '_tokens_to_score'):
                result = self.tokenizer._tokens_to_score([tok_seq])
            else:
                result = self.tokenizer.tokens_to_score([tok_seq])
            
            if result is not None:
                track_count = len(getattr(result, 'tracks', []))
                note_count = sum(len(t.notes) for t in getattr(result, 'tracks', []))
                print(f"  策略1结果: {track_count}个轨道, {note_count}个音符")
                if track_count > 0 and note_count > 0:
                    return result
            else:
                print("  策略1: 返回None")
        except Exception as e:
            print(f"  策略1异常: {e}")
            result = None
        
        # Try strategy 2: with default programs
        try:
            print("  尝试策略2: programs=[(0, False)]")
            if hasattr(self.tokenizer, '_tokens_to_score'):
                result = self.tokenizer._tokens_to_score([tok_seq], programs=[(0, False)])
            else:
                result = self.tokenizer.tokens_to_score([tok_seq], programs=[(0, False)])
            
            if result is not None:
                track_count = len(getattr(result, 'tracks', []))
                note_count = sum(len(t.notes) for t in getattr(result, 'tracks', []))
                print(f"  策略2结果: {track_count}个轨道, {note_count}个音符")
                if track_count > 0 and note_count > 0:
                    return result
            else:
                print("  策略2: 返回None")
        except Exception as e:
            print(f"  策略2异常: {e}")
            
        print("  ❌ PerTok内部解码失败")
        return None
    
    def _decode_strategy_pertok_architecture(self, tokens: List[int]) -> Optional[Score]:
        """
        使用PerTok架构进行手动解码
        这不是回退方案，而是真正的PerTok解码，严格遵循PerTok的token语义和架构
        """
        print("🏗️  PerTok架构解码 - 严格遵循PerTok token语义")
        
        # 清理和预处理tokens
        cleaned: List[int] = []
        vocab_size = max(self.vocab_reverse.keys()) + 1 if self.vocab_reverse else 0
        for t in tokens:
            if isinstance(t, int) and 0 <= t < vocab_size:
                cleaned.append(t)
        
        # PerTok架构要求的结构化处理
        cleaned = self._deduplicate_structure_tokens(cleaned)
        
        # 确保PerTok必需的结构tokens
        bos = self.vocab.get('BOS_None')
        eos = self.vocab.get('EOS_None')
        timesig = self.vocab.get('TimeSig_4/4')
        
        if bos is not None and (len(cleaned) == 0 or cleaned[0] != bos):
            cleaned.insert(0, bos)
        
        if timesig is not None:
            has_timesig = False
            for i in range(min(5, len(cleaned))):
                token_str = self.vocab_reverse.get(cleaned[i], '')
                if token_str.startswith('TimeSig_'):
                    has_timesig = True
                    break
            if not has_timesig:
                insert_pos = 1 if bos is not None else 0
                cleaned.insert(insert_pos, timesig)
        
        if eos is not None and (len(cleaned) == 0 or cleaned[-1] != eos):
            cleaned.append(eos)
        
        print(f"  PerTok结构化预处理完成: {len(cleaned)}个tokens")
        
        # 创建Score对象，使用PerTok的TPQ
        score = Score()
        score.ticks_per_quarter = self.tpq
        track = Track(program=0, is_drum=False, name="PerTokArchitecture")
        
        # PerTok架构的状态机解码
        current_time_ticks = 0
        current_velocity = 80
        pending_micro_shift_ticks = 0
        notes = []
        
        print(f"  开始PerTok状态机解码...")
        
        i = 0
        while i < len(cleaned):
            token_id = cleaned[i]
            token_str = self.vocab_reverse.get(token_id, f'UNK_{token_id}')
            
            # 跳过结构性tokens
            if token_str in ['BOS_None', 'EOS_None'] or token_str.startswith('TimeSig_'):
                i += 1
                continue
            
            # PerTok TimeShift处理 - 直接使用token中编码的绝对时间值
            if token_str.startswith('TimeShift_'):
                # PerTok格式: TimeShift_<beats>.<ticks>.<tpq>
                time_value = self._extract_pertok_time(token_str, 'TimeShift_')
                if time_value is not None:
                    current_time_ticks += int(time_value * self.tpq)
                i += 1
                continue
            
            # PerTok MicroTiming处理
            if token_str.startswith('MicroTiming_'):
                bin_val = self._extract_int(token_str, prefix='MicroTiming_')
                if bin_val is not None and self.num_micro_bins > 0 and self.max_micro_shift_beats > 0:
                    max_bin = self.num_micro_bins - 1
                    ratio = max(-1.0, min(1.0, (bin_val - max_bin/2) / (max_bin/2)))
                    shift_beats = ratio * self.max_micro_shift_beats
                    pending_micro_shift_ticks = int(shift_beats * self.tpq)
                i += 1
                continue
            
            # PerTok Velocity处理
            if token_str.startswith('Velocity_'):
                v = self._extract_int(token_str, prefix='Velocity_')
                if v is not None:
                    current_velocity = max(1, min(int(v), 127))
                i += 1
                continue
            
            # PerTok Pitch处理 - 核心音符创建
            if token_str.startswith('Pitch_'):
                pitch = self._extract_int(token_str, prefix='Pitch_')
                if pitch is not None:
                    # 默认参数
                    duration_ticks = self.tpq // 4  # 默认四分音符
                    note_velocity = current_velocity
                    note_micro_shift = pending_micro_shift_ticks

                    # 查找随后的 Velocity / MicroTiming / Duration（消费它们，避免二次推进时间）
                    consumed_until = i
                    look_end = min(i + 8, len(cleaned))
                    for j in range(i + 1, look_end):
                        next_token_str = self.vocab_reverse.get(cleaned[j], '')
                        if next_token_str.startswith('Velocity_') and note_velocity == current_velocity:
                            v = self._extract_int(next_token_str, prefix='Velocity_')
                            if v is not None:
                                note_velocity = max(1, min(int(v), 127))
                            consumed_until = max(consumed_until, j)
                            continue
                        if next_token_str.startswith('MicroTiming_') and note_micro_shift == pending_micro_shift_ticks:
                            bin_val = self._extract_int(next_token_str, prefix='MicroTiming_')
                            if bin_val is not None and self.num_micro_bins > 0 and self.max_micro_shift_beats > 0:
                                max_bin = self.num_micro_bins - 1
                                ratio = max(-1.0, min(1.0, (bin_val - max_bin/2) / (max_bin/2)))
                                shift_beats = ratio * self.max_micro_shift_beats
                                note_micro_shift = int(shift_beats * self.tpq)
                            consumed_until = max(consumed_until, j)
                            continue
                        if next_token_str.startswith('Duration_'):
                            duration_value = self._extract_pertok_time(next_token_str, 'Duration_')
                            if duration_value is not None:
                                duration_ticks = max(1, int(duration_value * self.tpq))
                            consumed_until = max(consumed_until, j)
                            break
                        if next_token_str.startswith('Pitch_'):
                            # 下一个Pitch，停止查找
                            break

                    # 创建音符（PerTok架构的核心）
                    start_ticks = current_time_ticks + note_micro_shift
                    note = Note(
                        time=max(0, start_ticks),
                        duration=duration_ticks,
                        pitch=int(pitch),
                        velocity=note_velocity
                    )
                    notes.append(note)

                    # 按PerTok语义：时间推进仅由后续的TimeShift决定，Duration只决定音符长度
                    pending_micro_shift_ticks = 0

                    # 跳过我们已消费的 Velocity/MicroTiming/Duration（保留TimeShift由主循环处理）
                    i = consumed_until + 1
                    continue
                # pitch 解析失败则继续
                i += 1
                continue
            
            # PerTok Duration处理（如果不在Pitch后）
            if token_str.startswith('Duration_'):
                # 在PerTok架构中，Duration通常跟随Pitch，这里可能是独立的
                duration_value = self._extract_pertok_time(token_str, 'Duration_')
                if duration_value is not None:
                    # 可能表示休止符或时间前进
                    current_time_ticks += int(duration_value * self.tpq)
                i += 1
                continue
            
            # 跳过其他token
            i += 1
        
        # 完成PerTok架构解码
        if len(notes) > 0:
            track.notes = notes
            score.tracks = [track]
            print(f"  ✅ PerTok架构解码完成: {len(notes)}个音符")
            return score
        else:
            print(f"  ❌ PerTok架构解码失败: 未生成音符")
            return None
    
    def _extract_pertok_time(self, token_str: str, prefix: str) -> Optional[float]:
        """
        提取PerTok时间值
        PerTok格式: TimeShift_<beats>.<ticks>.<tpq> 或 Duration_<beats>.<ticks>.<tpq>
        """
        if not token_str.startswith(prefix):
            return None
        
        # 移除前缀
        value_part = token_str[len(prefix):]
        
        # PerTok格式可能是 "1.0.320" -> 1.0 beats
        # 或者简单的 "1.0" -> 1.0 beats
        parts = value_part.split('.')
        
        try:
            if len(parts) >= 2:
                # "1.0.320" -> 提取 "1.0"
                beats = float(f"{parts[0]}.{parts[1]}")
                return beats
            elif len(parts) == 1:
                # "1" -> 1.0
                return float(parts[0])
        except ValueError:
            pass
        
        # 回退到regex提取
        import re
        m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", value_part)
        try:
            return float(m.group(0)) if m else None
        except Exception:
            return None

    def _decode_strategy_manual(self, tokens: List[int]) -> Optional[Score]:
        print("⚠️  使用手动回退解码（已应用 MicroTiming 与 TPQ）")
        score = Score()
        score.ticks_per_quarter = self.tpq
        track = Track(program=0, is_drum=False, name="ManualDecoded")

        current_time_ticks = 0
        current_velocity = 64
        pending_micro_shift_ticks = 0

        notes: List[Note] = []
        i = 0
        while i < len(tokens):
            t_id = tokens[i]
            t_str = self.vocab_reverse.get(t_id, '')

            # TimeShift
            if t_id in self.token_types['time_shift']:
                shift_beats = self._extract_float(t_str, prefix='TimeShift_')
                if shift_beats is not None:
                    current_time_ticks += int(shift_beats * self.tpq)
                i += 1
                continue

            # MicroTiming (collect and apply to next note)
            if t_id in self.token_types['microtiming']:
                bin_val = self._extract_int(t_str, prefix='MicroTiming_')
                if bin_val is not None and self.num_micro_bins > 0 and self.max_micro_shift_beats > 0:
                    # Map bin to [-1, 1]
                    max_bin = max(1, self.num_micro_bins // 2)
                    ratio = max(-1.0, min(1.0, float(bin_val) / float(max_bin)))
                    shift_beats = ratio * self.max_micro_shift_beats
                    pending_micro_shift_ticks = int(shift_beats * self.tpq)
                i += 1
                continue

            # Velocity
            if t_id in self.token_types['velocity']:
                v = self._extract_int(t_str, prefix='Velocity_')
                if v is not None:
                    current_velocity = max(1, min(int(v), 127))
                i += 1
                continue

            # Pitch -> create a note using nearest following Duration
            if t_id in self.token_types['pitch']:
                pitch = self._extract_int(t_str, prefix='Pitch_')
                duration_ticks = self.tpq // 4  # default to quarter note
                # Look ahead up to 6 tokens for Duration
                for j in range(i + 1, min(i + 7, len(tokens))):
                    nxt_id = tokens[j]
                    if nxt_id in self.token_types['duration']:
                        d_beats = self._extract_float(self.vocab_reverse.get(nxt_id, ''), prefix='Duration_')
                        if d_beats is not None:
                            duration_ticks = max(1, int(d_beats * self.tpq))
                            break
                start_ticks = current_time_ticks + pending_micro_shift_ticks
                pending_micro_shift_ticks = 0
                if pitch is not None:
                    notes.append(
                        Note(time=max(0, start_ticks), duration=duration_ticks, pitch=int(pitch), velocity=current_velocity)
                    )
                    # advance time by duration for monophonic melody
                    current_time_ticks += duration_ticks
                i += 1
                continue

            # Other tokens: ignore
            i += 1

        track.notes = notes
        score.tracks = [track]
        # Basic musical context
        score.tempos = [Tempo(time=0, mspq=500000)]  # 120 BPM
        score.time_signatures = [TimeSignature(time=0, numerator=4, denominator=4)]

        return score if len(notes) > 0 else None

    # ============== Helpers ==============
    def _extract_float(self, token_str: str, prefix: str) -> Optional[float]:
        # Be robust to tokens like "TimeShift_0.80.320" → extract first float
        if not token_str.startswith(prefix):
            return None
        m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", token_str[len(prefix):])
        try:
            return float(m.group(0)) if m else None
        except Exception:
            return None

    def _extract_int(self, token_str: str, prefix: str) -> Optional[int]:
        if not token_str.startswith(prefix):
            return None
        m = re.search(r"[-+]?\d+", token_str[len(prefix):])
        try:
            return int(m.group(0)) if m else None
        except Exception:
            return None
    
    def _deduplicate_structure_tokens(self, tokens: List[int]) -> List[int]:
        """去除重复的结构性tokens（BOS, TimeSig, EOS）"""
        cleaned = []
        seen_bos = False
        seen_timesig = False
        
        bos = self.vocab.get('BOS_None')
        eos = self.vocab.get('EOS_None')
        
        for token_id in tokens:
            token_str = self.vocab_reverse.get(token_id, '')
            
            # 跳过重复的BOS
            if token_id == bos:
                if not seen_bos:
                    cleaned.append(token_id)
                    seen_bos = True
                continue
            
            # 只保留第一个TimeSig
            if token_str.startswith('TimeSig_'):
                if not seen_timesig:
                    cleaned.append(token_id)
                    seen_timesig = True
                continue
            
            # EOS只保留在最后
            if token_id == eos:
                continue  # 稍后统一添加
            
            cleaned.append(token_id)
        
        return cleaned
    
    def _fix_malformed_tokens(self, tokens: List[int]) -> List[int]:
        """修复格式错误的tokens（如 Duration_1.0.320 -> Duration_1.0）"""
        fixed = []
        fixed_count = 0
        
        for token_id in tokens:
            token_str = self.vocab_reverse.get(token_id, '')
            
            # 检查是否是格式错误的token（包含多个点或异常格式）
            if any(prefix in token_str for prefix in ['Duration_', 'TimeShift_', 'MicroTiming_']) and token_str.count('.') > 1:
                # 提取第一个合理的数字部分
                if token_str.startswith('Duration_'):
                    match = re.search(r'Duration_([-+]?\d*\.?\d+)', token_str)
                    if match:
                        clean_value = match.group(1)
                        new_token_str = f'Duration_{clean_value}'
                        new_token_id = self.vocab.get(new_token_str, token_id)
                        if new_token_id != token_id:
                            fixed.append(new_token_id)
                            fixed_count += 1
                            continue
                
                elif token_str.startswith('TimeShift_'):
                    match = re.search(r'TimeShift_([-+]?\d*\.?\d+)', token_str)
                    if match:
                        clean_value = match.group(1)
                        new_token_str = f'TimeShift_{clean_value}'
                        new_token_id = self.vocab.get(new_token_str, token_id)
                        if new_token_id != token_id:
                            fixed.append(new_token_id)
                            fixed_count += 1
                            continue
                
                elif token_str.startswith('MicroTiming_'):
                    match = re.search(r'MicroTiming_([-+]?\d+)', token_str)
                    if match:
                        clean_value = match.group(1)
                        new_token_str = f'MicroTiming_{clean_value}'
                        new_token_id = self.vocab.get(new_token_str, token_id)
                        if new_token_id != token_id:
                            fixed.append(new_token_id)
                            fixed_count += 1
                            continue
            
            # 保持原token
            fixed.append(token_id)
        
        if fixed_count > 0:
            print(f"  修复了{fixed_count}个格式错误的tokens")
        
        return fixed


def create_fixed_decoder(tokenizer_config_func=None) -> FixedPerTokDecoder:
    if tokenizer_config_func is None:
        from working_pertok_config import create_working_config
        config = create_working_config()
    else:
        config = tokenizer_config_func()
    tokenizer = PerTok(config)
    return FixedPerTokDecoder(tokenizer)
