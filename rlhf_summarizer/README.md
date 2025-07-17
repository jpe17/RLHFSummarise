# RLHF Summarizer

This directory contains the core RLHF (Reinforcement Learning from Human Feedback) summarization scripts.

## Overview

The RLHF summarizer provides the main summarization functionality using advanced language models trained with reinforcement learning:

- **Baseline Summarizer**: Standard summarization pipeline
- **PPO Summarizer**: PPO-trained model for improved summaries
- **Command Line Interface**: Easy-to-use CLI for summarization tasks

## Files

### Core Scripts
- `run_tweet_summarizer.py` - Baseline tweet summarizer CLI
- `run_tweet_summarizer_ppo.py` - PPO-trained tweet summarizer CLI
- `README.md` - This file

## Usage

### Baseline Summarizer
```bash
# Basic usage
python run_tweet_summarizer.py username

# With options
python run_tweet_summarizer.py username --count 15 --save --max-length 200

# Save results
python run_tweet_summarizer.py username --save --output results.json

# Use specific device
python run_tweet_summarizer.py username --device cuda
```

### PPO Summarizer
```bash
# Basic usage
python run_tweet_summarizer_ppo.py username

# With options
python run_tweet_summarizer_ppo.py username --count 15 --save --max-length 200

# Compare with baseline
python run_tweet_summarizer_ppo.py username --compare-baseline
```

## Features

### Baseline Summarizer
- Standard transformer-based summarization
- Configurable summary length
- Multiple output formats
- Device selection (CPU/GPU)

### PPO Summarizer
- PPO-trained model for improved quality
- Reward-based scoring
- Comparison with baseline model
- Enhanced summary coherence

## Integration

The RLHF summarizer integrates with:
- `pipeline_twitter/` - For Twitter data processing
- `pipeline_instagram/` - For future Instagram data processing
- `backend/` - Core ML models and training logic
- Main web application (`app.py`) - For web interface

## Dependencies

- Core ML models from `backend/`
- PyTorch and transformers
- Twitter scraping components
- Voice synthesis (for integrated pipelines)

## Development Status

✅ **Fully implemented** - Both baseline and PPO summarizers are functional.

The RLHF summarizer is the core component of the content analysis system. 