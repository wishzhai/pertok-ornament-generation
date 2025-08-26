#!/usr/bin/env python3
"""
分析生成的token序列，特别关注TimeShift token的分布
"""

import torch
from working_pertok_config import create_working_tokenizer
from fixed_pertok_decoder import FixedPerTokDecoder
from inference import OrnamentInferenceEngine


def analyze_token_sequence(tokens, tokenizer, decoder):
    """详细分析token序列"""
    vocab = tokenizer.vocab
    id_to_str = {v: k for k, v in vocab.items()}

    print("=" * 80)
    print("🔍 TOKEN序列详细分析")
    print("=" * 80)

    # 统计各类token
    token_counts = {
        'pitch': 0,
        'velocity': 0,
        'duration': 0,
        'timeshift': 0,
        'microtiming': 0,
        'timesig': 0,
        'special': 0,
        'other': 0
    }

    timeshift_values = []
    duration_values = []

    print("\n📊 Token序列分析:")
    print(f"{'位置':<4} {'Token ID':<8} {'Token类型':<15} {'值':<20} {'描述'}")
    print("-" * 80)

    for i, token_id in enumerate(tokens):
        token_str = id_to_str.get(token_id, f'UNK_{token_id}')

        # 分类token
        if token_str.startswith('Pitch_'):
            token_counts['pitch'] += 1
            pitch_val = token_str.replace('Pitch_', '')
            print(f"{i:<4} {token_id:<8} {'Pitch':<15} {pitch_val:<20} MIDI音符")
        elif token_str.startswith('Velocity_'):
            token_counts['velocity'] += 1
            vel_val = token_str.replace('Velocity_', '')
            print(f"{i:<4} {token_id:<8} {'Velocity':<15} {vel_val:<20} 力度")
        elif token_str.startswith('Duration_'):
            token_counts['duration'] += 1
            dur_val = token_str.replace('Duration_', '')
            duration_values.append(dur_val)
            print(f"{i:<4} {token_id:<8} {'Duration':<15} {dur_val:<20} 时值")
        elif token_str.startswith('TimeShift_'):
            token_counts['timeshift'] += 1
            shift_val = token_str.replace('TimeShift_', '')
            timeshift_values.append(shift_val)
            print(f"{i:<4} {token_id:<8} {'TimeShift':<15} {shift_val:<20} 时间偏移")
        elif token_str.startswith('MicroTiming_'):
            token_counts['microtiming'] += 1
            micro_val = token_str.replace('MicroTiming_', '')
            print(f"{i:<4} {token_id:<8} {'MicroTiming':<15} {micro_val:<20} 微时序")
        elif token_str.startswith('TimeSig_'):
            token_counts['timesig'] += 1
            print(f"{i:<4} {token_id:<8} {'TimeSig':<15} {token_str.replace('TimeSig_', ''):<20} 拍号")
        elif token_str.endswith('_None'):
            token_counts['special'] += 1
            print(f"{i:<4} {token_id:<8} {'Special':<15} {token_str:<20} 特殊token")
        else:
            token_counts['other'] += 1
            print(f"{i:<4} {token_id:<8} {'Other':<15} {token_str:<20} 其他")

    print("\n" + "=" * 80)
    print("📈 TOKEN统计")
    print("=" * 80)

    total_tokens = len(tokens)
    for token_type, count in token_counts.items():
        percentage = (count / total_tokens) * 100 if total_tokens > 0 else 0
        print(f"{token_type.upper():<15}: {count:<4} ({percentage:>5.1f}%)")

    print(f"\n总token数: {total_tokens}")

    # 分析TimeShift分布
    if timeshift_values:
        print("\n" + "=" * 80)
        print("⏱️  TIMESHIFT详细分析")
        print("=" * 80)

        # 转换时间值为浮点数
        shift_floats = []
        for val in timeshift_values:
            try:
                # 处理PerTok格式 "1.0.320" -> 提取 "1.0"
                if '.' in val:
                    parts = val.split('.')
                    if len(parts) >= 2:
                        shift_floats.append(float(f"{parts[0]}.{parts[1]}"))
                    else:
                        shift_floats.append(float(val))
                else:
                    shift_floats.append(float(val))
            except Exception:
                continue

        if shift_floats:
            print("TimeShift值分布:")
            print(f"  最小值: {min(shift_floats):.3f} beats")
            print(f"  最大值: {max(shift_floats):.3f} beats")
            print(f"  平均值: {sum(shift_floats)/len(shift_floats):.3f} beats")

            # 分析停顿问题
            short_shifts = [s for s in shift_floats if s < 0.25]  # 小于1/4拍
            medium_shifts = [s for s in shift_floats if 0.25 <= s < 1.0]  # 1/4拍到1拍
            long_shifts = [s for s in shift_floats if s >= 1.0]  # 大于等于1拍

            print("\n停顿分析:")
            print(f"  短停顿 (<0.25拍): {len(short_shifts)} 个 ({len(short_shifts)/len(shift_floats)*100:.1f}%)")
            print(f"  中等停顿 (0.25-1拍): {len(medium_shifts)} 个 ({len(medium_shifts)/len(shift_floats)*100:.1f}%)")
            print(f"  长停顿 (≥1拍): {len(long_shifts)} 个 ({len(long_shifts)/len(shift_floats)*100:.1f}%)")

            if short_shifts:
                print(f"  短停顿值: {[f'{s:.3f}' for s in short_shifts[:10]]}")
            if long_shifts:
                print(f"  长停顿值: {[f'{s:.3f}' for s in long_shifts[:10]]}")

    # 分析Duration分布
    if duration_values:
        print("\n" + "=" * 80)
        print("🎵 DURATION详细分析")
        print("=" * 80)

        dur_floats = []
        for val in duration_values:
            try:
                if '.' in val:
                    parts = val.split('.')
                    if len(parts) >= 2:
                        dur_floats.append(float(f"{parts[0]}.{parts[1]}"))
                    else:
                        dur_floats.append(float(val))
                else:
                    dur_floats.append(float(val))
            except Exception:
                continue

        if dur_floats:
            print("Duration值分布:")
            print(f"  最小值: {min(dur_floats):.3f} beats")
            print(f"  最大值: {max(dur_floats):.3f} beats")
            print(f"  平均值: {sum(dur_floats)/len(dur_floats):.3f} beats")

            short_durs = [d for d in dur_floats if d < 0.5]  # 小于半拍
            medium_durs = [d for d in dur_floats if 0.5 <= d < 1.0]  # 半拍到1拍
            long_durs = [d for d in dur_floats if d >= 1.0]  # 大于等于1拍

            print("\n时值分析:")
            print(f"  短时值 (<0.5拍): {len(short_durs)} 个 ({len(short_durs)/len(dur_floats)*100:.1f}%)")
            print(f"  中等时值 (0.5-1拍): {len(medium_durs)} 个 ({len(medium_durs)/len(dur_floats)*100:.1f}%)")
            print(f"  长时值 (≥1拍): {len(long_durs)} 个 ({len(long_durs)/len(dur_floats)*100:.1f}%)")


def main():
    """主函数"""
    print("🔍 开始分析token序列...")

    # 初始化组件
    tokenizer = create_working_tokenizer()
    decoder = FixedPerTokDecoder(tokenizer)

    # 尝试加载模型（如果存在）
    model_path = 'checkpoints_ornament_aware/best_ornament_aware_model.pth'

    try:
        engine = OrnamentInferenceEngine(model_path, device='cpu')

        # 编码输入MIDI
        print("\n1️⃣ 编码输入MIDI...")
        input_tokens = engine.encode_midi('demo_input.mid')
        if input_tokens:
            print(f"✅ 输入编码成功: {len(input_tokens)} tokens")

            # 分析输入序列
            print("\n📊 分析输入序列:")
            analyze_token_sequence(input_tokens, tokenizer, decoder)

            # 生成装饰音
            print("\n2️⃣ 生成装饰音...")
            generated_tokens = engine.generate_ornaments(
                input_tokens,
                temperature=1.0,
                top_k=50,
                top_p=0.9
            )

            if generated_tokens:
                print(f"✅ 生成成功: {len(generated_tokens)} tokens")

                # 分析生成序列
                print("\n📊 分析生成序列:")
                analyze_token_sequence(generated_tokens, tokenizer, decoder)

                # 对比分析
                print("\n" + "=" * 80)
                print("🔄 输入vs输出对比")
                print("=" * 80)

                vocab = tokenizer.vocab
                id_to_str = {v: k for k, v in vocab.items()}

                input_timeshifts = sum(1 for t in input_tokens if id_to_str.get(t, '').startswith('TimeShift_'))
                output_timeshifts = sum(1 for t in generated_tokens if id_to_str.get(t, '').startswith('TimeShift_'))

                print("TimeShift tokens:")
                print(f"  输入: {input_timeshifts} 个")
                print(f"  输出: {output_timeshifts} 个")
                print(f"  增加: {output_timeshifts - input_timeshifts} 个")

                if len(input_tokens) > 0 and len(generated_tokens) > 0:
                    input_ratio = input_timeshifts / len(input_tokens) * 100
                    output_ratio = output_timeshifts / len(generated_tokens) * 100
                    print(f"  输入比例: {input_ratio:.1f}%")
                    print(f"  输出比例: {output_ratio:.1f}%")
                    print(f"  比例变化: {output_ratio - input_ratio:+.1f}%")

        else:
            print("❌ 输入编码失败")

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


