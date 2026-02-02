# FunASR-Qwen4B Audio Adaptor

A trained audio adaptor that projects SenseVoice encoder output into Qwen3-4B embedding space for ASR + translation.

## Architecture

```
Audio (WAV) → SenseVoice Encoder → [512-dim] → Audio Adaptor → [2560-dim] → Qwen3-4B → Text
```

### AudioAdaptorQwen4B (207M parameters)

```python
import torch
import torch.nn as nn

class AudioAdaptorQwen4B(nn.Module):
    def __init__(self, input_dim=512, output_dim=2560, hidden_dim=2048, num_layers=4, num_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        return self.norm(x)
```

## Training Pipeline

### Phase 1: Contrastive Alignment (10 epochs)
- **Loss**: InfoNCE contrastive loss
- **Goal**: Align audio embeddings with text embeddings in Qwen3 space
- **Output**: `adaptor_cached.pt`

### Phase 2: ASR Cross-Entropy (3 epochs)
- **Loss**: Cross-entropy on text generation
- **Goal**: Enable actual text generation from audio
- **LR**: 1e-5 with gradient clipping (0.5)
- **Output**: `adaptor_phase2_final.pt` (best model)

## Model Weights

| File | Size | Description |
|------|------|-------------|
| `adaptor_phase2_final.pt` | 793 MB | Best model (Phase 2 trained) |
| `adaptor_cached.pt` | 793 MB | Phase 1 model (contrastive only) |

## Dependencies

```bash
# requirements.txt
torch>=2.0
transformers>=4.40
funasr>=1.0
soundfile
tqdm
```

### Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers funasr soundfile tqdm
```

## Dataset Setup

### AISHELL-1 (178 hours Mandarin)

```bash
# Download from OpenSLR
wget https://www.openslr.org/resources/33/data_aishell.tgz
tar -xzf data_aishell.tgz

# Structure:
# data_aishell/
# ├── wav/
# │   ├── train/
# │   ├── dev/
# │   └── test/
# └── transcript/
#     └── aishell_transcript_v0.8.txt
```

## Inference Guide

### Basic ASR

```python
import torch
from funasr import AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import soundfile as sf

# Load models
device = "cuda"
sv_model = AutoModel(model="iic/SenseVoiceSmall", device=device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype=torch.float16, device_map="cuda")

# Load adaptor
adaptor = AudioAdaptorQwen4B().to(device)
ckpt = torch.load("adaptor_phase2_final.pt", map_location="cpu")
adaptor.load_state_dict({k: v.float() for k, v in ckpt.items()})
adaptor.eval()

# Hook to extract encoder output
encoder_output = None
def hook_fn(module, input, output):
    global encoder_output
    encoder_output = output[0] if isinstance(output, tuple) else output
hook = sv_model.model.encoder.register_forward_hook(hook_fn)

# Inference
audio, sr = sf.read("audio.wav")
_ = sv_model.generate(input="audio.wav", language="zh")
audio_feat = encoder_output.clone().to(device)
proj = adaptor(audio_feat).half()

# Generate with context
CONTEXT = "这是一段中文语音，请转录："
prompt_ids = tokenizer.encode(CONTEXT, return_tensors="pt").to(device)
prompt_embeds = qwen.model.embed_tokens(prompt_ids)
combined = torch.cat([prompt_embeds, proj], dim=1)

outputs = qwen.generate(
    inputs_embeds=combined,
    max_new_tokens=100,
    repetition_penalty=1.2,
    no_repeat_ngram_size=4
)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)

hook.remove()
```

### With Translation

```python
# After getting Chinese text, translate:
messages = [{"role": "user", "content": f"Translate to English: {cn_text}"}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # Disable thinking mode!
)
```

## Qwen3 Tips

### Disable Thinking Mode

```python
# Method 1: In chat template
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    enable_thinking=False  # Key parameter
)

# Method 2: Soft switch in prompt
prompt = "Your question here /no_think"
```

### Generation Parameters

```python
outputs = model.generate(
    inputs_embeds=combined,
    max_new_tokens=100,
    do_sample=False,
    repetition_penalty=1.2,      # Reduce repetition
    no_repeat_ngram_size=4,      # Block 4-gram repeats
)
```

## Training Scripts

| Script | Purpose |
|--------|---------|
| `train_phase2_asr.py` | Phase 2 ASR training |
| `train_cached.py` | Phase 1 contrastive training |
| `precompute_embeddings.py` | Precompute SenseVoice embeddings |
| `parallel_sensevoice.py` | Parallel preprocessing (12 workers) |

## New Server Setup

```bash
# 1. Extract backup
tar -xvf funasr-backup.tar

# 2. Create environment
python3 -m venv venv && source venv/bin/activate
pip install torch transformers funasr soundfile tqdm

# 3. Download dataset
wget https://www.openslr.org/resources/33/data_aishell.tgz
tar -xzf data_aishell.tgz

# 4. Precompute embeddings
python training/precompute_embeddings.py --data-dir data/data_aishell --cache-dir cached_embeddings

# 5. Continue training (optional)
python training/train_phase2_asr.py
```

## Performance

- **VRAM**: ~10-12 GB during inference
- **RTF**: ~0.007 (140x faster than real-time)
- **Training**: ~80 min for Phase 1 (10 epochs), ~90 min for Phase 2 (3 epochs) on GH200

## License

Training data: AISHELL-1 (Apache 2.0)
Models: SenseVoice (Apache 2.0), Qwen3-4B (Qwen License)
