# Lambda Cloud GH200 Connection

## SSH Access

```bash
ssh -i ~/home/tensordock/tensordock ubuntu@192.222.51.76
```

## Server Details

- **Instance**: NVIDIA GH200 480GB
- **VRAM**: 97.8 GB HBM3
- **RAM**: 525 GB
- **User**: ubuntu

## Training Project Location

```bash
cd ~/funasr-qwen4b
source venv/bin/activate
```

## Useful Commands

```bash
# Check GPU utilization
nvidia-smi

# Check preprocessing progress
find ~/funasr-qwen4b/hybrid_cache/audio -name '*.pt' | wc -l

# Check running processes
ps aux | grep parallel_sensevoice

# Start parallel preprocessing (12 workers)
python training/parallel_sensevoice.py --data-dir data/data_aishell --cache-dir hybrid_cache --num-workers 12
```
