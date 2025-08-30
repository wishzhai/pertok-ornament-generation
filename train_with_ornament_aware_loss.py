#!/usr/bin/env python3


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from miditok import PerTok
from tqdm import tqdm
import numpy as np
import os
import time
import argparse
import matplotlib.pyplot as plt

# Import core modules
from ornament_model import OrnamentTransformer
from working_pertok_config import create_working_config
from ornament_aware_loss import OrnamentAwareLoss, create_ornament_aware_loss


class OrnamentDatasetWithAwareLoss:
    """Ornament dataset compatible with OrnamentAwareLoss"""
    
    def __init__(self, maestro_path: str, max_seq_len: int = 512, max_files: int = None):
        """
        Initialize dataset
        
        Args:
            maestro_path: Path to MAESTRO dataset
            max_seq_len: Maximum sequence length
            max_files: Limit number of files (for quick testing)
        """
        self.maestro_path = Path(maestro_path)
        self.max_seq_len = max_seq_len
        
        # Create tokenizer with correct configuration
        config = create_working_config()
        self.tokenizer = PerTok(config)
        print(f"✅ Tokenizer created successfully, vocabulary size: {len(self.tokenizer.vocab)}")
        
        # Load MAESTRO metadata and filter for four composers
        csv_path = self.maestro_path / 'maestro-v3.0.0.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"MAESTRO CSV file not found: {csv_path}")
        
        self.dataset = pd.read_csv(csv_path)
        
        # Filter for four canonical composers (consistent with paper description)
        canonical_composers = [
            'Johann Sebastian Bach',
            'Wolfgang Amadeus Mozart', 
            'Ludwig van Beethoven',
            'Frédéric Chopin'
        ]
        
        self.dataset = self.dataset[
            self.dataset['canonical_composer'].isin(canonical_composers)
        ].copy()
        
        print(f"📊 Selected {len(self.dataset)} pieces ({len(self.dataset)/1276*100:.1f}% of MAESTRO)")
        print("Composer distribution:")
        for composer, count in self.dataset['canonical_composer'].value_counts().items():
            print(f"  {composer}: {count} pieces")
        
        # Use MAESTRO official data splits
        print("\n📋 Using MAESTRO official data splits:")
        print(f"  Training set: {len(self.dataset[self.dataset['split'] == 'train'])} pieces")
        print(f"  Validation set: {len(self.dataset[self.dataset['split'] == 'validation'])} pieces")
        print(f"  Test set: {len(self.dataset[self.dataset['split'] == 'test'])} pieces")
        
        if max_files:
            self.dataset = self.dataset.head(max_files)
            print(f"⚠️  Limited to first {max_files} files for quick testing")
            
        self.training_pairs = []
        self.validation_pairs = []
        self._prepare_training_data()
        
    def _prepare_training_data(self):
        """Prepare training data - create simplified→ornamented paired data"""
        print("🔄 Preparing training and validation data...")
        
        # Process training and validation sets separately
        train_data = self.dataset[self.dataset['split'] == 'train']
        val_data = self.dataset[self.dataset['split'] == 'validation']
        
        # Process training data
        print("Processing training set...")
        for _, row in tqdm(train_data.iterrows(), total=len(train_data), desc="Processing training MIDI files"):
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
                            
                            # Create multiple training pairs (sliding window)
                            for start_idx in range(0, len(original_tokens), self.max_seq_len // 2):
                                end_idx = min(start_idx + self.max_seq_len, len(original_tokens))
                                if end_idx - start_idx < 100:  # Skip sequences that are too short
                                    continue
                                    
                                segment = original_tokens[start_idx:end_idx]
                                
                                # Create simplified version
                                simplified = self._create_simplified_version(segment)
                                
                                if simplified is not None and len(simplified) > 50:
                                    # Ensure consistent sequence length
                                    input_tokens = simplified[:self.max_seq_len]
                                    target_tokens = segment[:self.max_seq_len]
                                    
                                    # Pad to same length
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
                print(f"⚠️  Error processing file {row.get('midi_filename', 'unknown')}: {e}")
                continue
        
        # Process validation data
        print("Processing validation set...")
        for _, row in tqdm(val_data.iterrows(), total=len(val_data), desc="Processing validation MIDI files"):
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
                            
                            # Create multiple validation pairs (sliding window)
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
                                    
                                    self.validation_pairs.append({
                                        'input_ids': input_tokens,
                                        'target_ids': target_tokens,
                                        'source': f"{row['midi_filename']}_track_{track_idx}_seg_{start_idx}"
                                    })
                                    
            except Exception as e:
                print(f"⚠️  Error processing validation file {row.get('midi_filename', 'unknown')}: {e}")
                continue
        
        print(f"✅ Generated training pairs: {len(self.training_pairs)}")
        print(f"✅ Generated validation pairs: {len(self.validation_pairs)}")
        
    def _create_simplified_version(self, tokens, removal_ratio=0.2):
        """Create simplified version - remove some ornament-related tokens"""
        if len(tokens) == 0:
            return None
        
        # Calculate token frequency
        from collections import Counter
        token_freq = Counter(tokens)
        
        # Identify potential ornament tokens
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
    
    def get_train_dataset(self):
        """获取训练数据集"""
        return OrnamentTrainDataset(self.training_pairs)
    
    def get_val_dataset(self):
        """获取验证数据集"""
        return OrnamentTrainDataset(self.validation_pairs)
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        item = self.training_pairs[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'target_ids': torch.tensor(item['target_ids'], dtype=torch.long),
            'attention_mask': torch.ones(len(item['input_ids']), dtype=torch.long)
        }


class OrnamentTrainDataset:
    """训练数据集包装器"""
    
    def __init__(self, pairs):
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        item = self.pairs[idx]
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
    """Train model using OrnamentAwareLoss"""
    print("=" * 70)
    print("🎵 Training ornament generation model with OrnamentAwareLoss")
    print("📄 Implementation fully consistent with paper workflow diagram")
    print("=" * 70)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
    
    # Create dataset
    print(f"\n📁 Loading dataset: {args.data_path}")
    dataset = OrnamentDatasetWithAwareLoss(
        maestro_path=args.data_path,
        max_seq_len=args.max_seq_len,
        max_files=args.max_files
    )
    
    if len(dataset) == 0:
        print("❌ Dataset is empty, please check data path")
        return
    
    # Use MAESTRO official splits
    train_dataset = dataset.get_train_dataset()
    val_dataset = dataset.get_val_dataset()
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"📊 Training set size: {len(train_dataset)}")
    print(f"📊 Validation set size: {len(val_dataset)}")
    print(f"📊 Batch size: {args.batch_size}")
    
    # Create model
    vocab_size = len(dataset.tokenizer.vocab)
    
    if args.resume:
        print(f"\n🔄 Resuming training from checkpoint: {args.resume}")
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
        print(f"✅ Model restored successfully, continuing from epoch {start_epoch}")
    else:
        print(f"\n🏗️  Creating new model")
        model = OrnamentTransformer(
            vocab_size=vocab_size,
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers
        ).to(device)
        start_epoch = 0
    
    print(f"✅ Model created successfully, vocabulary size: {vocab_size}")
    print(f"📏 Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create OrnamentAwareLoss - consistent with paper diagram
    print(f"\n🎯 Creating OrnamentAwareLoss")
    criterion = create_ornament_aware_loss(
        tokenizer=dataset.tokenizer,
        base_weight=1.0,
        ornament_boost=args.ornament_boost,
        new_content_boost=args.new_content_boost
    )
    
    # Optimizer and scheduler
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
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
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
        
        # Get ornament statistics
        stats = criterion.get_loss_statistics(target_ids)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Training loss: {avg_train_loss:.4f}")
        print(f"   Validation loss: {avg_val_loss:.4f}")
        print(f"   Learning rate: {current_lr:.2e}")
        print(f"   Ornament tokens: {stats['ornament_tokens']}/{stats['non_padding_tokens']} ({stats['ornament_tokens']/max(stats['non_padding_tokens'],1)*100:.1f}%)")
        print(f"   Average weight: {stats['average_weight']:.2f}")
        print(f"   Time elapsed: {epoch_time:.1f}s")
        
        # Save best model
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
            print(f"💾 Saved best model: {best_model_path}")
        
        # Save checkpoint periodically
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
            print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"🛑 Early stopping triggered (epoch {epoch+1})")
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
    print(f"📊 Training history plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train ornament generation model using OrnamentAwareLoss')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='maestro-v3.0.0-midi/maestro-v3.0.0',
                        help='Path to MAESTRO dataset')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Limit number of files (for quick testing)')
    
    # Model parameters
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=8,
                        help='Number of Transformer layers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=8,
                        help='Early stopping patience')
    
    # OrnamentAwareLoss参数
    parser.add_argument('--ornament_boost', type=float, default=2.5,
                        help='Ornament token weight multiplier')
    parser.add_argument('--new_content_boost', type=float, default=1.5,
                        help='New content weight multiplier')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Print configuration
    print("🔧 Training configuration:")
    for key, value in vars(args).items():
        print(f"   {key}: {value}")
    
    # Start training
    best_model_path = train_with_ornament_aware_loss(args)
    print(f"\n✅ Training completed, best model: {best_model_path}")


if __name__ == "__main__":
    main()






