#!/usr/bin/env python3
"""
Ornament Generation Model
基于PerTok和Transformer的装饰音生成模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import random
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import pandas as pd

from miditok import PerTok, TokenizerConfig
import pretty_midi

class OrnamentDataset(Dataset):
    """装饰音训练数据集"""
    
    def __init__(self, tokenizer: PerTok, maestro_path: str, max_seq_len: int = 512):
        """
        初始化数据集
        
        Args:
            tokenizer: PerTok tokenizer
            maestro_path: MAESTRO数据集路径
            max_seq_len: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.maestro_path = Path(maestro_path)
        
        # 加载MAESTRO元数据
        csv_path = self.maestro_path / "maestro-v3.0.0.csv"
        self.maestro_df = pd.read_csv(csv_path)
        
        # 筛选古典作品
        classical_composers = [
            'Johann Sebastian Bach', 
            'Wolfgang Amadeus Mozart', 
            'Ludwig van Beethoven',
            'Frédéric Chopin'
        ]
        
        self.classical_pieces = self.maestro_df[
            self.maestro_df['canonical_composer'].isin(classical_composers)
        ].copy()
        
        print(f"数据集初始化完成: {len(self.classical_pieces)} 首古典作品")
        
        # 预处理数据
        self.training_pairs = []
        self._prepare_training_data()
    
    def _prepare_training_data(self):
        """准备训练数据 - 修复PerTok返回类型处理"""
        print("准备训练数据...")
        
        for _, row in tqdm(self.classical_pieces.iterrows(), total=len(self.classical_pieces), desc="处理MIDI文件"):
            try:
                midi_path = Path(self.maestro_path) / row['midi_filename']
                if not midi_path.exists():
                    continue
                
                # 使用PerTok tokenizer
                tokenized_result = self.tokenizer(str(midi_path))
                print(f"  文件 {row['midi_filename']}: 返回类型 {type(tokenized_result)}")
                
                # PerTok返回TokSequence对象列表
                if isinstance(tokenized_result, list):
                    print(f"    包含 {len(tokenized_result)} 个轨道")
                    
                    for track_idx, tok_sequence in enumerate(tokenized_result):
                        if hasattr(tok_sequence, 'ids'):
                            original_token_ids = tok_sequence.ids
                            print(f"    轨道 {track_idx}: {len(original_token_ids)} 个tokens")
                            
                            if len(original_token_ids) > self.max_seq_len:
                                # 分割长序列
                                for i in range(0, len(original_token_ids) - self.max_seq_len, self.max_seq_len // 2):
                                    chunk = original_token_ids[i:i + self.max_seq_len]
                                    if len(chunk) >= 100:  # 最小长度要求
                                        simplified_chunk = self._create_simplified_version(chunk)
                                        if simplified_chunk is not None:
                                            self.training_pairs.append({
                                                'input': simplified_chunk,
                                                'target': chunk,
                                                'source_file': f"{row['midi_filename']}_track_{track_idx}"
                                            })
                            else:
                                if len(original_token_ids) >= 100:
                                    simplified_tokens = self._create_simplified_version(original_token_ids)
                                    if simplified_tokens is not None:
                                        self.training_pairs.append({
                                            'input': simplified_tokens,
                                            'target': original_token_ids,
                                            'source_file': f"{row['midi_filename']}_track_{track_idx}"
                                        })
                        else:
                            print(f"    轨道 {track_idx}: TokSequence没有ids属性")
                            
                elif hasattr(tokenized_result, 'ids'):
                    # 单个TokSequence对象
                    original_token_ids = tokenized_result.ids
                    print(f"    单轨道: {len(original_token_ids)} 个tokens")
                    
                    if len(original_token_ids) > self.max_seq_len:
                        # 分割长序列
                        for i in range(0, len(original_token_ids) - self.max_seq_len, self.max_seq_len // 2):
                            chunk = original_token_ids[i:i + self.max_seq_len]
                            if len(chunk) >= 100:
                                simplified_chunk = self._create_simplified_version(chunk)
                                if simplified_chunk is not None:
                                    self.training_pairs.append({
                                        'input': simplified_chunk,
                                        'target': chunk,
                                        'source_file': row['midi_filename']
                                    })
                    else:
                        if len(original_token_ids) >= 100:
                            simplified_tokens = self._create_simplified_version(original_token_ids)
                            if simplified_tokens is not None:
                                self.training_pairs.append({
                                    'input': simplified_tokens,
                                    'target': original_token_ids,
                                    'source_file': row['midi_filename']
                                })
                else:
                    print(f"    无法识别的返回类型: {type(tokenized_result)}")
                    
            except Exception as e:
                print(f"处理文件 {midi_path} 时出错: {e}")
                continue
        
        print(f"生成训练对: {len(self.training_pairs)} 个")
    
    def _create_simplified_version(self, token_ids: List[int]) -> Optional[List[int]]:
        """
        创建简化版本 - 专门针对PerTok的token结构
        PerTok序列：TimeShift → Pitch → Velocity(可选) → MicroTiming(可选) → Duration(可选)
        
        简化策略：移除MicroTiming tokens和部分短Duration
        """
        try:
            if not hasattr(self.tokenizer, 'vocab'):
                # 如果无法访问词汇表，使用采样简化
                return self._random_simplify(token_ids)
            
            vocab = self.tokenizer.vocab
            id_to_token = {v: k for k, v in vocab.items()}
            
            simplified_tokens = []
            i = 0
            removed_count = 0
            
            while i < len(token_ids):
                token_id = token_ids[i]
                
                # 获取token字符串
                token_str = id_to_token.get(token_id, "")
                
                # PerTok特定的简化规则
                should_remove = False
                
                # 1. 移除MicroTiming tokens（装饰音的核心特征）
                if "MicroTiming" in token_str:
                    should_remove = True
                    
                # 2. 移除非常短的Duration tokens（快速装饰音）
                elif "Duration" in token_str and "0." in token_str:
                    # 例如：Duration_0.1、Duration_0.2等
                    try:
                        # 提取duration值
                        duration_part = token_str.split("_")[1] if "_" in token_str else ""
                        if duration_part and float(duration_part) < 0.5:  # 小于半拍的duration
                            should_remove = True
                    except:
                        pass
                        
                # 3. 移除部分很短的TimeShift（装饰音之间的微小间隔）
                elif "TimeShift" in token_str and "0." in token_str:
                    try:
                        shift_part = token_str.split("_")[1] if "_" in token_str else ""
                        if shift_part and float(shift_part) < 0.25:  # 小于1/4拍的timeshift
                            if removed_count < len(token_ids) * 0.1:  # 限制移除数量
                                should_remove = True
                    except:
                        pass
                
                if should_remove:
                    removed_count += 1
                else:
                    simplified_tokens.append(token_id)
                    
                i += 1
            
            # 检查简化效果
            removal_ratio = removed_count / len(token_ids) if len(token_ids) > 0 else 0
            
            if removal_ratio >= 0.05 and len(simplified_tokens) >= 50:  # 至少移除5%
                print(f"  规则简化：{len(token_ids)} → {len(simplified_tokens)} (移除 {removal_ratio:.1%})")
                return simplified_tokens
            else:
                # 规则简化效果不明显，使用随机简化
                return self._random_simplify(token_ids)
                
        except Exception as e:
            print(f"  简化失败: {e}")
            return self._random_simplify(token_ids)
    
    def _random_simplify(self, token_ids: List[int]) -> List[int]:
        """随机简化：保留75-85%的tokens，模拟移除装饰音"""
        import random
        
        keep_ratio = random.uniform(0.75, 0.85)
        keep_count = int(len(token_ids) * keep_ratio)
        
        if keep_count < 50:  # 确保最小长度
            keep_count = min(50, len(token_ids))
        
        # 随机选择要保留的位置，但保持顺序
        indices = sorted(random.sample(range(len(token_ids)), keep_count))
        simplified_tokens = [token_ids[i] for i in indices]
        
        removal_ratio = (len(token_ids) - len(simplified_tokens)) / len(token_ids)
        print(f"  随机简化：{len(token_ids)} → {len(simplified_tokens)} (移除 {removal_ratio:.1%})")
        
        return simplified_tokens
    
    def _simplify_midi(self, midi):
        """
        简化MIDI文件，移除可能的装饰音
        
        Args:
            midi: PrettyMIDI对象
            
        Returns:
            简化后的MIDI
        """
        simplified_midi = pretty_midi.PrettyMIDI()
        
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
                
            simplified_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=False
            )
            
            notes = sorted(instrument.notes, key=lambda x: x.start)
            filtered_notes = []
            
            for note in notes:
                duration = note.end - note.start
                
                # 过滤条件：移除可能的装饰音
                if (duration >= 0.2 and  # 保留较长的音符
                    note.velocity >= 50):  # 保留有一定力度的音符
                    
                    # 检查是否为快速连续音符（可能的装饰音）
                    is_ornament = False
                    for other_note in notes:
                        if (other_note != note and
                            abs(other_note.start - note.end) < 0.1 and
                            abs(other_note.pitch - note.pitch) <= 3 and
                            other_note.end - other_note.start < 0.15):
                            is_ornament = True
                            break
                    
                    if not is_ornament:
                        filtered_notes.append(note)
            
            simplified_instrument.notes = filtered_notes
            simplified_midi.instruments.append(simplified_instrument)
        
        return simplified_midi
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        pair = self.training_pairs[idx]
        
        # 转换为tensor
        input_ids = torch.tensor(pair['input'][:self.max_seq_len], dtype=torch.long)
        target_ids = torch.tensor(pair['target'][:self.max_seq_len], dtype=torch.long)
        
        # Padding
        if len(input_ids) < self.max_seq_len:
            pad_length = self.max_seq_len - len(input_ids)
            input_ids = F.pad(input_ids, (0, pad_length), value=0)  # 使用0作为padding
        
        if len(target_ids) < self.max_seq_len:
            pad_length = self.max_seq_len - len(target_ids)
            target_ids = F.pad(target_ids, (0, pad_length), value=0)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': (input_ids != 0).long()
        }

class OrnamentTransformer(nn.Module):
    """装饰音生成Transformer模型"""
    
    def __init__(self, vocab_size: int, max_seq_len: int = 512, d_model: int = 256, n_heads: int = 8, n_layers: int = 6):
        """
        初始化模型
        
        Args:
            vocab_size: 词汇表大小
            max_seq_len: 最大序列长度
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: 层数
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Embedding层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播
        
        Args:
            input_ids: 输入token序列 [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            logits: 输出logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Position encoding
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        embeddings = self.dropout(token_embeds + pos_embeds)
        
        # Attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (True for positions to mask)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        hidden_states = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        return logits
    
    def generate_ornaments(self, input_ids, max_length=None, temperature=1.0, top_k=50):
        """
        生成装饰音
        
        Args:
            input_ids: 输入序列
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: top-k采样
            
        Returns:
            生成的序列
        """
        self.eval()
        
        if max_length is None:
            max_length = input_ids.shape[1]
        
        with torch.no_grad():
            # 获取模型输出
            logits = self.forward(input_ids)
            
            # 应用温度
            logits = logits / temperature
            
            # Top-k采样
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(top_k_logits, dim=-1)
                
                # 采样
                sampled_indices = torch.multinomial(probs.view(-1, top_k), 1)
                generated_tokens = top_k_indices.view(-1, top_k).gather(1, sampled_indices).view(logits.shape[:2])
            else:
                # 贪心解码
                generated_tokens = torch.argmax(logits, dim=-1)
        
        return generated_tokens

class OrnamentTrainer:
    """装饰音模型训练器"""
    
    def __init__(self, model: OrnamentTransformer, tokenizer: PerTok, device: torch.device):
        """
        初始化训练器
        
        Args:
            model: 装饰音模型
            tokenizer: PerTok tokenizer
            device: 计算设备
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=1e-4, 
            weight_decay=0.01
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding token
    
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # 前向传播
            logits = self.model(input_ids, attention_mask)
            
            # 计算损失
            loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        return total_loss / num_batches
    
    def save_model(self, save_path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab_size': self.model.vocab_size,
            'max_seq_len': self.model.max_seq_len,
            'd_model': self.model.d_model
        }, save_path)
        print(f"模型已保存到: {save_path}")
    
    def load_model(self, load_path: str):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"模型已从 {load_path} 加载")

def train_ornament_model():
    """训练装饰音模型的主函数"""
    print("=== 开始训练装饰音生成模型 ===")
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化PerTok tokenizer
    config = TokenizerConfig(
        pitch_range=(21, 108),
        beat_res={(0, 4): 8, (0, 4): 12},
        use_chords=False,
        use_programs=False,
        use_microtiming=True,
        ticks_per_quarter=480,
        max_microtiming_shift=0.125,
        num_microtiming_bins=32,
        use_tempos=True,
        use_time_signatures=True
    )
    
    tokenizer = PerTok(config)
    print("PerTok tokenizer 初始化完成")
    
    # 创建数据集
    dataset = OrnamentDataset(
        tokenizer=tokenizer,
        maestro_path="maestro-v3.0.0-midi/maestro-v3.0.0",
        max_seq_len=256  # 较小的序列长度用于快速训练
    )
    
    if len(dataset) == 0:
        print("❌ 没有生成训练数据，请检查MAESTRO数据集路径")
        return
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # 小批次用于快速训练
        shuffle=True,
        num_workers=0  # Windows兼容性
    )
    
    # 创建模型
    model = OrnamentTransformer(
        vocab_size=len(tokenizer.vocab),
        max_seq_len=256,
        d_model=128,  # 较小的模型用于快速训练
        n_heads=4,
        n_layers=4
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = OrnamentTrainer(model, tokenizer, device)
    
    # 训练
    num_epochs = 3  # 快速训练
    print(f"开始训练 {num_epochs} 个epochs...")
    
    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch(dataloader, epoch + 1)
        print(f"Epoch {epoch + 1}/{num_epochs}, 平均损失: {avg_loss:.4f}")
    
    # 保存模型
    model_save_path = "ornament_model.pth"
    trainer.save_model(model_save_path)
    
    print("✅ 训练完成！")
    return trainer, tokenizer

if __name__ == "__main__":
    trainer, tokenizer = train_ornament_model() 