#!/usr/bin/env python3
"""
工作的PerTok配置
自动生成于 fix_pertok_config.py
"""

from miditok import PerTok, TokenizerConfig

def create_working_config():
    """创建工作的PerTok配置"""
    TOKENIZER_PARAMS = {'pitch_range': (21, 109), 'beat_res': {(0, 4): 4, (4, 12): 3}, 'special_tokens': ['PAD', 'BOS', 'EOS', 'MASK'], 'use_chords': False, 'use_rests': False, 'use_tempos': False, 'use_time_signatures': True, 'use_programs': False, 'use_microtiming': True, 'ticks_per_quarter': 320, 'max_microtiming_shift': 0.125, 'num_microtiming_bins': 30}
    
    return TokenizerConfig(**TOKENIZER_PARAMS)

def create_working_tokenizer():
    """创建工作的PerTok tokenizer"""
    config = create_working_config()
    return PerTok(config)

if __name__ == "__main__":
    tokenizer = create_working_tokenizer()
    print(f"工作配置词汇表大小: {len(tokenizer.vocab)}")
