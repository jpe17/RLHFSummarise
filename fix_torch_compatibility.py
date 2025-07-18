#!/usr/bin/env python3
import torch
import torch.serialization

def fix_torch_tts_compatibility():
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        torch.serialization.add_safe_globals([XttsConfig])
        print("✅ Added TTS config to PyTorch safe globals")
        return True
    except ImportError:
        print("⚠️ TTS not available, skipping compatibility fix")
        return False
    except Exception as e:
        print(f"⚠️ Could not fix TTS compatibility: {e}")
        return False

fix_torch_tts_compatibility()
