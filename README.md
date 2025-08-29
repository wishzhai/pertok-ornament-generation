# Leveraging PerTok and Domain-Specific Transformer Design for Expressive MIDI Ornament Generation

This repository contains the implementation of our paper on expressive MIDI ornament generation using PerTok tokenization and domain-specific Transformer architecture.

## 🎵 Overview

Our approach introduces a novel framework for generating expressive musical ornaments by:
- **PerTok tokenization** with MicroTiming support for fine-grained musical representation (320 TPQ)
- **Domain-specific Transformer** architecture optimized for musical sequence generation
- **Ornament-Aware Loss Function** that specifically weights ornament-related tokens during training
- **Curated MAESTRO subset** of 530 pieces from 4 canonical composers spanning Baroque to Romantic periods
- **Robust PerTok-compliant decoding** that ensures generated sequences follow PerTok architecture principles

## 🔬 Key Contributions

1. **PerTok Architecture Decoder**: A robust decoder that strictly follows PerTok token semantics and architecture
2. **Ornament-Aware Loss Function**: Dynamic weighting of ornament tokens (MicroTiming, short durations, high velocities) 
3. **Curated Training Dataset**: 530-piece subset focusing on ornament-rich classical repertoire
4. **Expression Enhancement**: Achieves 2-3% increase in ornament density while maintaining musical coherence

## 📊 Dataset

We curated a subset of **530 pieces** from 4 canonical composers representing **41.5%** of the MAESTRO dataset:

- **Johann Sebastian Bach**: 145 pieces (Baroque ornamental traditions)
- **Wolfgang Amadeus Mozart**: 38 pieces (Classical elegance)
- **Ludwig van Beethoven**: 146 pieces (Classical-Romantic transition)
- **Frédéric Chopin**: 201 pieces (Romantic expressiveness)

This focused selection ensures stylistic coherence while covering core ornamental practices in Western classical music.

## 🏗️ Architecture

### PerTok Configuration
- **Pitch Range**: 21-109 (88 keys)
- **Ticks Per Quarter**: 320
- **MicroTiming**: Enabled with 30 bins (±0.125 shift)
- **Beat Resolution**: Variable (4 for 0-4 beats, 3 for 4-12 beats)
- **Vocabulary Size**: ~304 tokens

### Transformer Model
- **Layers**: 8
- **Attention Heads**: 8
- **Model Dimension**: 512
- **Maximum Sequence Length**: 512 tokens
- **Parameters**: ~25M

### Ornament-Aware Loss
- **Base Loss**: CrossEntropyLoss
- **Ornament Token Boost**: 2.5×
  - MicroTiming tokens
  - Short Duration tokens (<0.5 beats)
  - Short TimeShift tokens (<0.25 beats)
  - High Velocity tokens
- **New Content Boost**: 1.5×
  - Tokens present in target but not input (generated ornaments)

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create conda environment (use the same name as in scripts)
conda create -n orna python=3.9
conda activate orna

# Install dependencies
pip install torch torchvision torchaudio
pip install miditok symusic pandas numpy matplotlib tqdm
```

### 2. Data Preparation
```bash
# Download MAESTRO v3.0.0 dataset
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip maestro-v3.0.0-midi.zip
```

### 3. Training
```bash
# Train with Ornament-Aware Loss (recommended)
python train_with_ornament_aware_loss.py \
    --epochs 20 \
    --batch_size 6 \
    --ornament_boost 2.5 \
    --new_content_boost 1.5

# Training will automatically use the curated 530-piece subset
```

### 4. Inference
```bash
# Generate ornaments for a MIDI file
python inference.py \
    --input examples/input.mid \
    --output results/output_with_ornaments.mid \
    --temperature 1.1 \
    --top_k 50

# Use specific model
python inference.py \
    --model checkpoints_ornament_aware/best_ornament_aware_model.pth \
    --input input.mid \
    --output output.mid
```

## 📁 Directory Structure

```
ornamentsscore/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── 🎵 Core Components
├── working_pertok_config.py               # PerTok tokenizer configuration
├── ornament_model.py                      # OrnamentTransformer & Dataset
├── fixed_pertok_decoder.py                # Robust MIDI decoding
├── ornament_aware_loss.py                 # Ornament-weighted loss function
│
├── 🚀 Training & Inference
├── train_with_ornament_aware_loss.py      # Main training script
├── inference.py                           # Unified inference CLI
├── improved_trainer.py                    # Legacy trainer (reference)
│
├── 🧪 Testing & Examples
├── generate_ornaments.py                  # Quick testing script
├── complete_ornament_generation.py        # Full pipeline demo
│
├── 📊 Data & Models
├── maestro-v3.0.0-midi/                  # MAESTRO dataset
├── checkpoints_ornament_aware/            # OrnamentAware models
├── checkpoints/                           # Legacy models
├── examples/                              # Example MIDI files
└── results/                               # Generated outputs
```

## 🎯 Key Features

### 1. PerTok Architecture Decoding
Our `FixedPerTokDecoder` implements true PerTok architecture compliance:
- **Primary Strategy**: PerTok internal decoding with enhanced error handling
- **PerTok Architecture Decoder**: Manual decoder that strictly follows PerTok token semantics and state machine
- **Fallback Strategy**: Traditional manual decoding as last resort
- **Token Format Support**: Correctly handles PerTok's `Duration_1.0.320` and `TimeShift_1.0.320` formats
- **MicroTiming Integration**: Full support for PerTok's 30-bin microtiming system

### 2. Ornament-Aware Training
The `OrnamentAwareLoss` automatically identifies and weights ornament tokens:
```python
# Example usage
from ornament_aware_loss import create_ornament_aware_loss

criterion = create_ornament_aware_loss(
    tokenizer=tokenizer,
    ornament_boost=2.5,    # 2.5× weight for ornament tokens
    new_content_boost=1.5   # 1.5× weight for newly generated content
)
```

### 3. Flexible Inference
The unified inference script supports various generation parameters:
```bash
python inference.py \
    --input input.mid \
    --output output.mid \
    --temperature 1.1      # Controls randomness (0.5-2.0)
    --top_k 50            # Top-k sampling
    --top_p 0.9           # Nucleus sampling
    --device cuda         # Force GPU/CPU
```

## 📈 Training Progress

The training script provides comprehensive monitoring:
- Real-time loss tracking (train/validation)
- Ornament token statistics
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Training history visualization

## 🎼 Model Performance

### PerTok Architecture Compliance
- **True PerTok Decoding**: Achieves 96+ musical notes from 481 generated tokens
- **Token Format Support**: Correctly interprets `Duration_1.0.320` format with 320 TPQ
- **MicroTiming Resolution**: 31 unique microtiming bins with ±0.125 beat shift
- **State Machine Accuracy**: Maintains proper Pitch→Duration→TimeShift sequences

### Generation Quality Metrics
- **Ornament Enhancement**: 2-3% increase in ornament density (21.1% → 23.3%)
- **Musical Coherence**: Preserves harmonic progression and rhythmic structure  
- **Expression Quality**: Generated ~100 notes with microtiming variations
- **PerTok Compliance**: 100% adherence to PerTok token semantics and architecture

## 🏗️ PerTok Architecture Implementation

### Token Format Specification
Our implementation strictly follows PerTok's token format:
```
Duration_1.0.320    # 1.0 beats at 320 TPQ
TimeShift_1.0.320   # 1.0 beat time advance at 320 TPQ  
MicroTiming_15      # Bin 15 of 30 microtiming bins
Pitch_60           # MIDI note 60 (Middle C)
Velocity_80        # MIDI velocity 80
```

### State Machine Decoding
The PerTok architecture decoder implements a stateful parser:
1. **Structure Tokens**: `BOS_None` → `TimeSig_4/4` → musical content → `EOS_None`
2. **Musical Events**: `Pitch` → (optional `Velocity`/`MicroTiming`) → `Duration` → `TimeShift`
3. **Time Tracking**: Absolute time maintenance with microtiming offsets
4. **Note Creation**: Precise timing with PerTok's 320 TPQ resolution

### Technical Validation
- ✅ **Token Parsing**: Correctly extracts numeric values from PerTok format strings
- ✅ **Time Calculation**: Accurate beat→tick conversion with 320 TPQ
- ✅ **MicroTiming**: Proper bin→offset mapping with ±0.125 beat range
- ✅ **Sequence Validation**: Maintains PerTok's grammar constraints

## 🔧 Advanced Usage

### Custom Training
```bash
# Fine-tune existing model
python train_with_ornament_aware_loss.py \
    --resume checkpoints_ornament_aware/best_ornament_aware_model.pth \
    --epochs 10 \
    --learning_rate 1e-5

# Adjust ornament weighting
python train_with_ornament_aware_loss.py \
    --ornament_boost 3.0 \
    --new_content_boost 2.0
```

### Batch Processing
```bash
# Process multiple files
for file in inputs/*.mid; do
    python inference.py --input "$file" --output "results/$(basename "$file")"
done
```

### Analysis Tools
```python
# Analyze ornament content (ornament ratio and average weight)
from ornament_aware_loss import OrnamentTokenAnalyzer
from working_pertok_config import create_working_tokenizer
import torch

tokenizer = create_working_tokenizer()
analyzer = OrnamentTokenAnalyzer(tokenizer)

# Tokenize
tok_seq_list = tokenizer("example.mid")
tokens = tok_seq_list[0].ids if isinstance(tok_seq_list, list) else tok_seq_list.ids

# Compute stats
total = len(tokens)
ornament_count = sum(1 for t in tokens if analyzer.is_ornament_token(t))
avg_weight = sum(analyzer.get_ornament_weight(t) for t in tokens) / max(total, 1)

print(f"Ornament density: {ornament_count/total:.1%}")
print(f"Average token weight: {avg_weight:.2f}")

# Or get per-category counts
from collections import Counter
cat_counts = Counter()
for t in tokens:
    for cat, ids in analyzer.ornament_tokens.items():
        if t in ids:
            cat_counts[cat] += 1
print("Category counts:", dict(cat_counts))
```

## 📋 Paper Implementation Status

### ✅ Fully Implemented
- [x] **PerTok Architecture Decoder** with true token semantics compliance
- [x] **Ornament-Aware Loss Function** with dynamic token weighting
- [x] **Domain-Specific Transformer** (8 layers, 8 heads, 512d, 25M params)
- [x] **Curated MAESTRO Dataset** (530 pieces, 4 composers)
- [x] **MicroTiming Integration** (30 bins, ±0.125 beat shift, 320 TPQ)
- [x] **Expression Enhancement** (2-3% ornament density increase)

### 🎯 Key Results Achieved
- **PerTok Compliance**: 100% adherence to PerTok architecture
- **Generation Quality**: 96+ notes from 481 tokens with microtiming
- **Ornament Enhancement**: 21.1% → 23.3% ornament density
- **Technical Validation**: Correct handling of `Duration_1.0.320` format

## 🤝 Citation

If you use this code in your research, please cite:

```bibtex

```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MidiTok](https://github.com/Natooz/MidiTok) for the PerTok tokenizer implementation
- [symusic](https://github.com/Yikai-Liao/symusic) for efficient MIDI processing
- [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) for high-quality piano performance data
- The classical composers whose works form the foundation of our ornamental understanding

## 🔗 Related Work

- [MidiTok: A Python package for MIDI file tokenization](https://github.com/Natooz/MidiTok)
- [The MAESTRO Dataset and Wave2Midi2Wave](https://arxiv.org/abs/1810.12247)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
