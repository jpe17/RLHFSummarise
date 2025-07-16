import torch

print("=== Comparing weight files ===")

# Load both files
w1 = torch.load('lora_weights.pt', map_location='cpu')
w2 = torch.load('simple_ppo_lora_final_20250716_130239.pt', map_location='cpu')

print(f"Keys in w1: {len(w1.keys())}")
print(f"Keys in w2: {len(w2.keys())}")
print(f"Same keys: {set(w1.keys()) == set(w2.keys())}")

# Check if files are identical
identical = True
differences = 0
for key in w1.keys():
    if not torch.equal(w1[key], w2[key]):
        print(f"Different at key: {key}")
        print(f"  w1 shape: {w1[key].shape}")
        print(f"  w2 shape: {w2[key].shape}")
        print(f"  w1 first few values: {w1[key].flatten()[:5]}")
        print(f"  w2 first few values: {w2[key].flatten()[:5]}")
        print(f"  Max difference: {torch.max(torch.abs(w1[key] - w2[key]))}")
        print(f"  Mean difference: {torch.mean(torch.abs(w1[key] - w2[key]))}")
        identical = False
        differences += 1
        if differences > 3:  # Only show first few differences
            break

print(f"Files are identical: {identical}")
print(f"Number of different keys: {differences}")

if identical:
    print("⚠️ WARNING: The files appear to be identical!")
    print("This suggests that either:")
    print("1. The PPO training didn't actually update the weights")
    print("2. The weights were copied/overwritten")
    print("3. Both files contain the same pre-PPO weights")
else:
    print("✅ The files are different - PPO training did update the weights") 