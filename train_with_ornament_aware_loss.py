#!/usr/bin/env python3
"""
使用OrnamentAwareLoss的完整训练脚本
与论文流程图完全一致的训练实现
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import pandas as pd
from miditok import PerTok
from tqdm import tqdm
import numpy as np
import os
import time
import argparse
import matplotlib.pyplot as plt

# 导入核心模块
from ornament_model import OrnamentTransformer
from working_pertok_config import create_working_config
from ornament_aware_loss import OrnamentAwareLoss, create_ornament_aware_loss


class OrnamentDatasetWithAwareLoss:
    """配合OrnamentAwareLoss的装饰音数据集"""
    
    def __init__(self, maestro_path: str, max_seq_len: int = 512, max_files: int = None):
        """
        初始化数据集
        
        Args:
            maestro_path: MAESTRO数据集路径
            max_seq_len: 最大序列长度
            max_files: 限制文件数量（用于快速测试）
        """
        self.maestro_path = Path(maestro_path)
        self.max_seq_len = max_seq_len
        
        # 使用正确配置创建tokenizer
        config = create_working_config()
        self.tokenizer = PerTok(config)
        print(f"✅ Tokenizer创建成功，词汇表大小: {len(self.tokenizer.vocab)}")
        
        # 加载MAESTRO元数据并筛选四位作曲家
        csv_path = self.maestro_path / 'maestro-v3.0.0.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"MAESTRO CSV文件不存在: {csv_path}")
        
        self.dataset = pd.read_csv(csv_path)
        
        # 筛选四位canonical作曲家（与论文描述一致）
        canonical_composers = [
            'Johann Sebastian Bach',
            'Wolfgang Amadeus Mozart', 
            'Ludwig van Beethoven',
            'Frédéric Chopin'
        ]
        
        self.dataset = self.dataset[
            self.dataset['canonical_composer'].isin(canonical_composers)
        ].copy()
        
        print(f"📊 筛选出{len(self.dataset)}首作品 (占MAESTRO的{len(self.dataset)/1276*100:.1f}%)")
        print("作曲家分布:")
        for composer, count in self.dataset['canonical_composer'].value_counts().items():
            print(f"  {composer}: {count}首")
        
        if max_files:
            self.dataset = self.dataset.head(max_files)
            print(f"⚠️  限制为前{max_files}个文件用于快速测试")
            
        self.training_pairs = []
        self._prepare_training_data()
        
    def _prepare_training_data(self):
        """准备训练数据 - 创建简化→装饰的配对数据"""
        print("🔄 准备训练数据...")
        
        for _, row in tqdm(self.dataset.iterrows(), total=len(self.dataset), desc="处理MIDI文件"):
            try:
                midi_path = self.maestro_path / row['midi_filename']
                if not midi_path.exists():
                    continue
                
                # 使用PerTok编码
                tokenized_result = self.tokenizer(str(midi_path))
                
                if isinstance(tokenized_result, list):
                    for track_idx, tok_sequence in enumerate(tokenized_result):
                        if hasattr(tok_sequence, 'ids'):
                            original_tokens = tok_sequence.ids
                            
                            # 创建多个训练对（滑动窗口）
                            for start_idx in range(0, len(original_tokens), self.max_seq_len // 2):
                                end_idx = min(start_idx + self.max_seq_len, len(original_tokens))
                                if end_idx - start_idx < 100:  # 跳过太短的序列
                                    continue
                                    
                                segment = original_tokens[start_idx:end_idx]
                                
                                # 创建简化版本
                                simplified = self._create_simplified_version(segment)
                                
                                if simplified is not None and len(simplified) > 50:
                                    # 确保序列长度一致
                                    input_tokens = simplified[:self.max_seq_len]
                                    target_tokens = segment[:self.max_seq_len]
                                    
                                    # 填充到相同长度
                                    if len(input_tokens) < self.max_seq_len:
                                        input_tokens.extend([0] * (self.max_seq_len - len(input_tokens)))
                                    if len(target_tokens) < self.max_seq_len:
                                        target_tokens.extend([0] * (self.max_seq_len - len(target_tokens)))
                                    
                                    self.training_pairs.append({
                                        'input_ids': input_tokens,
                                        'target_ids': target_tokens,
                                        'source': f"{row['midi_filename']}_track_{track_idx}_seg_{start_idx}"
                                    })
                                    
            except Exception as e:
                print(f"⚠️  处理文件时出错 {row.get('midi_filename', 'unknown')}: {e}")
                continue
        
        print(f"✅ 生成训练对数量: {len(self.training_pairs)}")
        
    def _create_simplified_version(self, tokens, removal_ratio=0.2):
        """创建简化版本 - 移除部分装饰音相关tokens"""
        if len(tokens) == 0:
            return None
        
        # 计算token频率
        from collections import Counter
        token_freq = Counter(tokens)
        
        # 识别可能的装饰音tokens
        ornament_candidates = set()
        
        # 罕见tokens更可能是装饰音
        rare_threshold = max(1, len(tokens) // 50)
        rare_tokens = set([token for token, freq in token_freq.items() if freq <= rare_threshold])
        ornament_candidates.update(rare_tokens)
        
        # 基于词汇表识别装饰音相关tokens
        vocab_reverse = {v: k for k, v in self.tokenizer.vocab.items()}
        for token_id in tokens:
            if token_id < len(vocab_reverse):
                token_str = vocab_reverse[token_id]
                
                # MicroTiming tokens
                if token_str.startswith('MicroTiming_'):
                    ornament_candidates.add(token_id)
                
                # 短时值tokens  
                elif token_str.startswith('Duration_'):
                    try:
                        duration_val = float(token_str.replace('Duration_', ''))
                        if duration_val < 0.5:  # 短于半拍
                            ornament_candidates.add(token_id)
                    except:
                        pass
                
                # 短时间偏移tokens
                elif token_str.startswith('TimeShift_'):
                    try:
                        shift_val = float(token_str.replace('TimeShift_', ''))
                        if shift_val < 0.25:  # 短于1/4拍
                            ornament_candidates.add(token_id)
                    except:
                        pass
        
        # 智能移除
        simplified = []
        removed_count = 0
        target_removal = int(len(tokens) * removal_ratio)
        
        for token in tokens:
            should_remove = False
            
            # 优先移除装饰音候选
            if (token in ornament_candidates and 
                removed_count < target_removal and
                token not in [0, 1, 2, 3]):  # 保留特殊tokens
                should_remove = True
                removed_count += 1
            
            if not should_remove:
                simplified.append(token)
        
        return simplified if len(simplified) >= 50 else None
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        item = self.training_pairs[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'target_ids': torch.tensor(item['target_ids'], dtype=torch.long),
            'attention_mask': torch.ones(len(item['input_ids']), dtype=torch.long)
        }


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 8, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


def train_with_ornament_aware_loss(args):
    """使用OrnamentAwareLoss训练模型"""
    print("=" * 70)
    print("🎵 使用OrnamentAwareLoss训练装饰音生成模型")
    print("📄 与论文流程图完全一致的实现")
    print("=" * 70)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
    
    # 创建数据集
    print(f"\n📁 加载数据集: {args.data_path}")
    dataset = OrnamentDatasetWithAwareLoss(
        maestro_path=args.data_path,
        max_seq_len=args.max_seq_len,
        max_files=args.max_files
    )
    
    if len(dataset) == 0:
        print("❌ 数据集为空，请检查数据路径")
        return
    
    # 创建训练验证集分割
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"📊 训练集大小: {len(train_dataset)}")
    print(f"📊 验证集大小: {len(val_dataset)}")
    print(f"📊 批量大小: {args.batch_size}")
    
    # 创建模型
    vocab_size = len(dataset.tokenizer.vocab)
    
    if args.resume:
        print(f"\n🔄 从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model = OrnamentTransformer(
            vocab_size=vocab_size,
            max_seq_len=args.max_seq_len,
            d_model=checkpoint.get('d_model', 512),
            n_heads=8,
            n_layers=8
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"✅ 模型恢复成功，从第{start_epoch}轮继续")
    else:
        print(f"\n🏗️  创建新模型")
        model = OrnamentTransformer(
            vocab_size=vocab_size,
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers
        ).to(device)
        start_epoch = 0
    
    print(f"✅ 模型创建成功，词汇表大小: {vocab_size}")
    print(f"📏 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建OrnamentAwareLoss - 与论文图一致
    print(f"\n🎯 创建OrnamentAwareLoss")
    criterion = create_ornament_aware_loss(
        tokenizer=dataset.tokenizer,
        base_weight=1.0,
        ornament_boost=args.ornament_boost,
        new_content_boost=args.new_content_boost
    )
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=len(train_loader) * args.epochs,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # 早停机制
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    
    # 创建检查点目录
    checkpoint_dir = Path("checkpoints_ornament_aware")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    learning_rates = []
    
    print(f"\n🚀 开始训练 {args.epochs} 个epochs...")
    start_time = time.time()
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_steps = 0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + args.epochs} 训练")
        for batch in train_progress:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # 使用OrnamentAwareLoss
            loss = criterion(outputs, target_ids, input_ids)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            # 更新进度条
            train_progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1} 验证")
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, target_ids, input_ids)
                
                val_loss += loss.item()
                val_steps += 1
                
                val_progress.set_postfix({
                    'val_loss': f'{loss.item():.4f}'
                })
        
        avg_val_loss = val_loss / val_steps
        val_losses.append(avg_val_loss)
        
        # 获取装饰音统计
        stats = criterion.get_loss_statistics(target_ids)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印epoch总结
        print(f"\n📊 Epoch {epoch+1} 总结:")
        print(f"   训练损失: {avg_train_loss:.4f}")
        print(f"   验证损失: {avg_val_loss:.4f}")
        print(f"   学习率: {current_lr:.2e}")
        print(f"   装饰音tokens: {stats['ornament_tokens']}/{stats['non_padding_tokens']} ({stats['ornament_tokens']/max(stats['non_padding_tokens'],1)*100:.1f}%)")
        print(f"   平均权重: {stats['average_weight']:.2f}")
        print(f"   耗时: {epoch_time:.1f}s")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = checkpoint_dir / "best_ornament_aware_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': create_working_config(),
                'vocab_size': vocab_size,
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'd_model': args.d_model,
                'n_heads': args.n_heads,
                'n_layers': args.n_layers,
                'max_seq_len': args.max_seq_len
            }, best_model_path)
            print(f"💾 保存最佳模型: {best_model_path}")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': create_working_config(),
                'vocab_size': vocab_size,
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'd_model': args.d_model,
                'n_heads': args.n_heads,
                'n_layers': args.n_layers,
                'max_seq_len': args.max_seq_len
            }, checkpoint_path)
            print(f"💾 保存检查点: {checkpoint_path}")
        
        # 早停检查
        if early_stopping(avg_val_loss):
            print(f"🛑 早停触发 (epoch {epoch+1})")
            break
        
        print("-" * 50)
    
    total_time = time.time() - start_time
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, learning_rates, checkpoint_dir)
    
    print(f"\n🎉 训练完成!")
    print(f"⏱️  总耗时: {total_time/3600:.1f}小时")
    print(f"🏆 最佳验证损失: {best_val_loss:.4f}")
    print(f"📁 模型保存在: {checkpoint_dir}")
    
    return best_model_path


def plot_training_history(train_losses, val_losses, learning_rates, save_dir):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('OrnamentAwareLoss 训练曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 学习率曲线
    ax2.plot(epochs, learning_rates, 'g-', label='学习率', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('学习率')
    ax2.set_title('学习率变化')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / 'ornament_aware_training_history.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 训练历史图已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='使用OrnamentAwareLoss训练装饰音生成模型')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='maestro-v3.0.0-midi/maestro-v3.0.0',
                        help='MAESTRO数据集路径')
    parser.add_argument('--max_files', type=int, default=None,
                        help='限制文件数量（用于快速测试）')
    
    # 模型参数
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--d_model', type=int, default=512,
                        help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=8,
                        help='Transformer层数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=8,
                        help='早停耐心')
    
    # OrnamentAwareLoss参数
    parser.add_argument('--ornament_boost', type=float, default=2.5,
                        help='装饰音token权重倍数')
    parser.add_argument('--new_content_boost', type=float, default=1.5,
                        help='新增内容权重倍数')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    # 打印配置
    print("🔧 训练配置:")
    for key, value in vars(args).items():
        print(f"   {key}: {value}")
    
    # 开始训练
    best_model_path = train_with_ornament_aware_loss(args)
    print(f"\n✅ 训练完成，最佳模型: {best_model_path}")


if __name__ == "__main__":
    main()
