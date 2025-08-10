#!/usr/bin/env python3
"""
装饰音感知损失函数
基于PerTok词汇表分析，准确识别装饰音相关tokens并应用专门的损失权重
"""

import torch
import torch.nn as nn
from typing import List, Dict, Set, Optional
from miditok import PerTok
import re
from collections import defaultdict


class OrnamentTokenAnalyzer:
    """装饰音token分析器"""
    
    def __init__(self, tokenizer: PerTok):
        """
        初始化分析器
        
        Args:
            tokenizer: PerTok tokenizer实例
        """
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}
        
        # 分析装饰音相关tokens
        self.ornament_tokens = self._identify_ornament_tokens()
        self.token_weights = self._calculate_token_weights()
        
    def _identify_ornament_tokens(self) -> Dict[str, Set[int]]:
        """识别装饰音相关的token集合"""
        ornament_tokens = {
            'microtiming': set(),      # 微时序tokens（装饰音的核心特征）
            'short_duration': set(),   # 短时值tokens
            'short_timeshift': set(),  # 短时间偏移tokens
            'high_velocity': set(),    # 高力度tokens（突出的装饰音）
            'grace_notes': set(),      # 装饰音特征组合
            'ornamental_pitches': set() # 装饰性音高模式
        }
        
        for token_str, token_id in self.vocab.items():
            # 1. 微时序tokens - 装饰音的核心特征
            if self._is_microtiming_token(token_str):
                ornament_tokens['microtiming'].add(token_id)
            
            # 2. 短时值tokens - 快速经过的装饰音
            elif self._is_short_duration_token(token_str):
                ornament_tokens['short_duration'].add(token_id)
            
            # 3. 短时间偏移tokens - 装饰音之间的微小间隔
            elif self._is_short_timeshift_token(token_str):
                ornament_tokens['short_timeshift'].add(token_id)
            
            # 4. 高力度tokens - 突出的装饰音
            elif self._is_high_velocity_token(token_str):
                ornament_tokens['high_velocity'].add(token_id)
        
        # 5. 组合特征 - 典型装饰音模式
        ornament_tokens['grace_notes'] = self._identify_grace_note_patterns()
        
        # 打印分析结果
        print("装饰音token分析结果:")
        total_ornament_tokens = 0
        for category, token_set in ornament_tokens.items():
            print(f"  {category}: {len(token_set)} tokens")
            total_ornament_tokens += len(token_set)
        
        print(f"总装饰音相关tokens: {total_ornament_tokens}/{len(self.vocab)} ({100*total_ornament_tokens/len(self.vocab):.1f}%)")
        
        return ornament_tokens
    
    def _is_microtiming_token(self, token_str: str) -> bool:
        """判断是否为微时序token"""
        if not token_str.startswith('MicroTiming_'):
            return False
        
        try:
            # 提取微时序值
            value_str = token_str.replace('MicroTiming_', '')
            value = float(value_str)
            
            # 所有微时序都是装饰音特征，但小的偏移更可能是装饰音
            return True  # 所有MicroTiming都算装饰音特征
        except ValueError:
            return False
    
    def _is_short_duration_token(self, token_str: str) -> bool:
        """判断是否为短时值token"""
        if not token_str.startswith('Duration_'):
            return False
        
        try:
            # 提取时值
            value_str = token_str.replace('Duration_', '')
            duration = float(value_str)
            
            # 小于半拍的时值更可能是装饰音
            return duration < 0.5
        except ValueError:
            return False
    
    def _is_short_timeshift_token(self, token_str: str) -> bool:
        """判断是否为短时间偏移token"""
        if not token_str.startswith('TimeShift_'):
            return False
        
        try:
            # 提取时间偏移值
            value_str = token_str.replace('TimeShift_', '')
            shift = float(value_str)
            
            # 小于1/4拍的偏移更可能与装饰音相关
            return shift < 0.25
        except ValueError:
            return False
    
    def _is_high_velocity_token(self, token_str: str) -> bool:
        """判断是否为高力度token"""
        if not token_str.startswith('Velocity_'):
            return False
        
        try:
            # 提取力度值
            value_str = token_str.replace('Velocity_', '')
            velocity = int(value_str)
            
            # 高力度(>90)可能是突出的装饰音
            return velocity > 90
        except ValueError:
            return False
    
    def _identify_grace_note_patterns(self) -> Set[int]:
        """识别典型装饰音模式的token组合"""
        grace_tokens = set()
        
        # 寻找极短时值的tokens（典型的装饰音特征）
        for token_str, token_id in self.vocab.items():
            if token_str.startswith('Duration_'):
                try:
                    value_str = token_str.replace('Duration_', '')
                    duration = float(value_str)
                    
                    # 极短时值（小于1/8拍）通常是装饰音
                    if duration < 0.125:
                        grace_tokens.add(token_id)
                except ValueError:
                    continue
        
        return grace_tokens
    
    def _calculate_token_weights(self) -> Dict[int, float]:
        """计算每个token的装饰音权重"""
        weights = {}
        
        # 基础权重为1.0
        for token_id in range(len(self.vocab)):
            weights[token_id] = 1.0
        
        # 为装饰音相关tokens分配更高权重
        weight_map = {
            'microtiming': 3.0,         # 微时序是最重要的装饰音特征
            'grace_notes': 2.5,         # 极短装饰音
            'short_duration': 2.0,      # 短时值
            'short_timeshift': 1.8,     # 短时间偏移
            'high_velocity': 1.5,       # 高力度
            'ornamental_pitches': 1.3   # 装饰性音高
        }
        
        for category, token_set in self.ornament_tokens.items():
            category_weight = weight_map.get(category, 1.0)
            for token_id in token_set:
                # 如果token属于多个类别，取最高权重
                weights[token_id] = max(weights.get(token_id, 1.0), category_weight)
        
        return weights
    
    def get_ornament_weight(self, token_id: int) -> float:
        """获取指定token的装饰音权重"""
        return self.token_weights.get(token_id, 1.0)
    
    def is_ornament_token(self, token_id: int) -> bool:
        """判断token是否为装饰音相关"""
        for token_set in self.ornament_tokens.values():
            if token_id in token_set:
                return True
        return False
    
    def get_ornament_statistics(self) -> Dict:
        """获取装饰音token统计信息"""
        stats = {
            'total_vocab_size': len(self.vocab),
            'ornament_categories': {},
            'weight_distribution': defaultdict(int)
        }
        
        # 统计各类别数量
        for category, token_set in self.ornament_tokens.items():
            stats['ornament_categories'][category] = len(token_set)
        
        # 统计权重分布
        for weight in self.token_weights.values():
            stats['weight_distribution'][f'{weight:.1f}'] += 1
        
        return stats


class OrnamentAwareLoss(nn.Module):
    """装饰音感知损失函数 - 完整实现版本"""
    
    def __init__(self, tokenizer: PerTok, base_weight: float = 1.0, 
                 ornament_boost: float = 2.0, new_content_boost: float = 1.5,
                 pad_token_id: int = 0):
        """
        初始化装饰音感知损失函数
        
        Args:
            tokenizer: PerTok tokenizer实例
            base_weight: 基础权重
            ornament_boost: 装饰音token额外权重倍数
            new_content_boost: 新增内容额外权重倍数
            pad_token_id: padding token ID
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.base_weight = base_weight
        self.ornament_boost = ornament_boost
        self.new_content_boost = new_content_boost
        self.pad_token_id = pad_token_id
        
        # 初始化装饰音token分析器
        self.analyzer = OrnamentTokenAnalyzer(tokenizer)
        
        # 基础交叉熵损失
        self.base_criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
        
        # 预计算权重张量（用于快速查找）
        self.register_buffer('token_weights', self._create_weight_tensor())
        
        print(f"装饰音感知损失函数初始化完成:")
        print(f"  词汇表大小: {len(tokenizer.vocab)}")
        print(f"  装饰音token数量: {sum(len(s) for s in self.analyzer.ornament_tokens.values())}")
        print(f"  基础权重: {base_weight}")
        print(f"  装饰音权重倍数: {ornament_boost}")
        print(f"  新增内容权重倍数: {new_content_boost}")
    
    def _create_weight_tensor(self) -> torch.Tensor:
        """创建权重查找张量"""
        vocab_size = len(self.tokenizer.vocab)
        weights = torch.ones(vocab_size, dtype=torch.float32)
        
        for token_id in range(vocab_size):
            base_ornament_weight = self.analyzer.get_ornament_weight(token_id)
            # 应用装饰音权重倍数
            if base_ornament_weight > 1.0:
                weights[token_id] = base_ornament_weight * self.ornament_boost
            else:
                weights[token_id] = self.base_weight
        
        return weights
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                input_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算装饰音感知损失
        
        Args:
            logits: 模型输出 [batch_size, seq_len, vocab_size]
            targets: 目标序列 [batch_size, seq_len]
            input_tokens: 输入序列 [batch_size, seq_len] (用于计算新增内容)
            
        Returns:
            加权损失值
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # 基础交叉熵损失
        base_loss = self.base_criterion(logits.view(-1, vocab_size), targets.view(-1))
        base_loss = base_loss.view(batch_size, seq_len)
        
        # 创建权重矩阵
        weights = torch.ones_like(targets, dtype=torch.float, device=device)
        
        # 1. 应用装饰音token权重
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                target_token = targets[batch_idx, seq_idx].item()
                if target_token != self.pad_token_id and target_token < len(self.token_weights):
                    weights[batch_idx, seq_idx] = self.token_weights[target_token]
        
        # 2. 对新增内容（相对于输入）额外加权
        if input_tokens is not None:
            new_content_mask = (targets != input_tokens) & (targets != self.pad_token_id)
            weights[new_content_mask] *= self.new_content_boost
        
        # 3. 对连续装饰音序列给予额外关注
        ornament_sequence_boost = self._detect_ornament_sequences(targets)
        weights *= ornament_sequence_boost
        
        # 应用权重
        weighted_loss = base_loss * weights
        
        # 只对非padding位置计算平均损失
        mask = (targets != self.pad_token_id)
        if mask.sum() > 0:
            return weighted_loss[mask].mean()
        else:
            return weighted_loss.mean()
    
    def _detect_ornament_sequences(self, targets: torch.Tensor) -> torch.Tensor:
        """检测装饰音序列并给予额外权重"""
        batch_size, seq_len = targets.shape
        sequence_weights = torch.ones_like(targets, dtype=torch.float, device=targets.device)
        
        for batch_idx in range(batch_size):
            ornament_count = 0
            for seq_idx in range(seq_len):
                token_id = targets[batch_idx, seq_idx].item()
                
                if self.analyzer.is_ornament_token(token_id):
                    ornament_count += 1
                    # 连续装饰音获得递增权重
                    if ornament_count >= 2:
                        boost = min(1.5, 1.0 + 0.1 * ornament_count)
                        sequence_weights[batch_idx, seq_idx] = boost
                else:
                    ornament_count = 0
        
        return sequence_weights
    
    def get_loss_statistics(self, targets: torch.Tensor) -> Dict:
        """获取损失计算统计信息"""
        stats = {
            'total_tokens': 0,
            'ornament_tokens': 0,
            'non_padding_tokens': 0,
            'average_weight': 0.0,
            'ornament_categories': defaultdict(int)
        }
        
        non_pad_mask = targets != self.pad_token_id
        stats['total_tokens'] = targets.numel()
        stats['non_padding_tokens'] = non_pad_mask.sum().item()
        
        if stats['non_padding_tokens'] > 0:
            # 计算装饰音token数量和平均权重
            total_weight = 0.0
            for batch_idx in range(targets.shape[0]):
                for seq_idx in range(targets.shape[1]):
                    if non_pad_mask[batch_idx, seq_idx]:
                        token_id = targets[batch_idx, seq_idx].item()
                        if token_id < len(self.token_weights):
                            weight = self.token_weights[token_id].item()
                            total_weight += weight
                            
                            if self.analyzer.is_ornament_token(token_id):
                                stats['ornament_tokens'] += 1
                                
                                # 统计装饰音类别
                                for category, token_set in self.analyzer.ornament_tokens.items():
                                    if token_id in token_set:
                                        stats['ornament_categories'][category] += 1
            
            stats['average_weight'] = total_weight / stats['non_padding_tokens']
        
        return stats


def create_ornament_aware_loss(tokenizer: PerTok, **kwargs) -> OrnamentAwareLoss:
    """创建装饰音感知损失函数的便捷函数"""
    return OrnamentAwareLoss(tokenizer, **kwargs)


if __name__ == "__main__":
    # 测试装饰音感知损失函数
    print("=== 测试装饰音感知损失函数 ===")
    
    from working_pertok_config import create_working_tokenizer
    
    # 创建tokenizer
    tokenizer = create_working_tokenizer()
    
    # 创建损失函数
    loss_fn = create_ornament_aware_loss(tokenizer)
    
    # 测试数据
    batch_size, seq_len, vocab_size = 2, 10, len(tokenizer.vocab)
    
    # 模拟logits和targets
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 计算损失
    loss = loss_fn(logits, targets, input_tokens)
    
    print(f"测试损失值: {loss.item():.4f}")
    
    # 获取统计信息
    stats = loss_fn.get_loss_statistics(targets)
    print("损失统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("✅ 装饰音感知损失函数测试完成!")
