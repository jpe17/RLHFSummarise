#!/usr/bin/env python3
"""
Simple PPO Training for LoRA Fine-tuning
Single file, sequential implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from tqdm import tqdm
from datetime import datetime

# Import existing components
from model import setup_lora_model, load_lora_weights, save_lora_weights
from reward import load_reward_model
from data_loader import setup_tokenizer

# Configuration
MODEL_ID = "Qwen/Qwen1.5-0.5B"
LORA_WEIGHTS_PATH = "../lora_weights.pt"
REWARD_MODEL_PATH = "../qwen_reward_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# PPO Hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
PPO_EPOCHS = 4
CLIP_EPS = 0.2
VALUE_COEF = 0.1
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 1.0
GAMMA = 0.99
LAM = 0.95

# Generation parameters
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_P = 0.9

print(f"🚀 Simple PPO Training")
print(f"Device: {DEVICE}")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"PPO Epochs: {PPO_EPOCHS}")
print(f"Clip Epsilon: {CLIP_EPS}")
print("="*50)

# 1. Setup models and tokenizer
print("📝 Setting up tokenizer...")
tokenizer = setup_tokenizer(MODEL_ID)

print("🤖 Loading policy model...")
policy_model = setup_lora_model(MODEL_ID, DEVICE)
if os.path.exists(LORA_WEIGHTS_PATH):
    policy_model = load_lora_weights(policy_model, LORA_WEIGHTS_PATH)
    print(f"✅ Loaded LoRA weights from {LORA_WEIGHTS_PATH}")
policy_model.train()

print("🏆 Loading reward model...")
reward_model, _ = load_reward_model(REWARD_MODEL_PATH, DEVICE)
reward_model.eval()

# 2. Setup value head (simple linear layer)
print("📈 Setting up value head...")
hidden_size = policy_model.config.hidden_size
value_head = nn.Linear(hidden_size, 1).to(DEVICE)

# 3. Setup optimizers
policy_optimizer = torch.optim.AdamW(
    [p for p in policy_model.parameters() if p.requires_grad],
    lr=LEARNING_RATE
)
value_optimizer = torch.optim.AdamW(value_head.parameters(), lr=LEARNING_RATE)

# 4. Load training data
print("📊 Loading training data...")
with open("../data/train.jsonl", 'r') as f:
    training_data = [json.loads(line) for line in f]

prompts = [item['prompt'] for item in training_data[:500]]  # Use subset for demo
print(f"Loaded {len(prompts)} prompts")

def generate_response(prompt):
    """Generate response using policy model"""
    full_prompt = f"Please summarize the following tweets:\n\n{prompt}\n\nSummary:"
    
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=400,
        padding=True
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = policy_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Extract response
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text[len(full_prompt):].strip()
    
    return response, outputs[0]

def get_reward(prompt, response):
    """Get reward from reward model"""
    text = f"Post: {prompt}\nSummary: {response}"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    ).to(DEVICE)
    
    with torch.no_grad():
        reward = reward_model(inputs["input_ids"], inputs["attention_mask"])
        return reward.item() if reward.dim() > 0 else float(reward)

def compute_log_prob_and_value(prompt, response, requires_grad=False):
    """Compute log probability and value for a response"""
    full_prompt = f"Please summarize the following tweets:\n\n{prompt}\n\nSummary:"
    full_text = f"{full_prompt} {response}"
    
    # Tokenize
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(DEVICE)
    
    # Forward pass - conditional gradient computation
    if requires_grad:
        outputs = policy_model(**inputs, return_dict=True, output_hidden_states=True)
    else:
        with torch.no_grad():
            outputs = policy_model(**inputs, return_dict=True, output_hidden_states=True)
        
    # Get log probabilities
    logits = outputs.logits[0]  # [seq_len, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    
    # Get response tokens
    prompt_tokens = tokenizer(full_prompt, return_tensors="pt").input_ids.to(DEVICE)
    prompt_len = prompt_tokens.shape[1]
    
    response_tokens = inputs.input_ids[0][prompt_len:inputs.input_ids.shape[1]-1]  # Exclude last token
    response_log_probs = []
    response_entropies = []
    
    for i, token_id in enumerate(response_tokens):
        if prompt_len + i < logits.shape[0]:
            token_log_prob = log_probs[prompt_len + i, token_id]
            response_log_probs.append(token_log_prob)
            
            # Compute entropy for this position
            entropy = -(probs[prompt_len + i] * log_probs[prompt_len + i]).sum()
            response_entropies.append(entropy)
    
    # Average log probability and entropy
    avg_log_prob = torch.stack(response_log_probs).mean() if response_log_probs else torch.tensor(0.0).to(DEVICE)
    avg_entropy = torch.stack(response_entropies).mean() if response_entropies else torch.tensor(0.0).to(DEVICE)
    
    # Value from last hidden state
    last_hidden = outputs.hidden_states[-1][0, -1, :]  # [hidden_size]
    value = value_head(last_hidden).squeeze()
    
    return avg_log_prob, value, avg_entropy

def compute_advantages(rewards, values, gamma=GAMMA, lam=LAM):
    """Compute advantages using GAE"""
    advantages = []
    returns = []
    
    next_value = 0
    next_advantage = 0
    
    for i in reversed(range(len(rewards))):
        # Compute return
        next_return = rewards[i] + gamma * next_value
        returns.insert(0, next_return)
        
        # Compute advantage
        delta = rewards[i] + gamma * next_value - values[i]
        next_advantage = delta + gamma * lam * next_advantage
        advantages.insert(0, next_advantage)
        
        next_value = values[i]
    
    # Normalize advantages
    advantages = torch.tensor(advantages, device=DEVICE)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages.tolist(), returns

# 5. Main training loop
print("\n🎯 Starting PPO Training...")
num_episodes = 20

for episode in range(num_episodes):
    print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
    
    # Step 1: Collect experiences
    print("🔄 Collecting experiences...")
    batch_prompts = np.random.choice(prompts, size=BATCH_SIZE, replace=False)
    
    experiences = []
    for prompt in tqdm(batch_prompts, desc="Generating"):
        response, _ = generate_response(prompt)
        
        if response.strip():  # Only keep non-empty responses
            reward = get_reward(prompt, response)
            log_prob, value, _ = compute_log_prob_and_value(prompt, response)  # Ignore entropy during data collection
            
            experiences.append({
                'prompt': prompt,
                'response': response,
                'reward': reward,
                'log_prob': log_prob.item(),
                'value': value.item()
            })
    
    if not experiences:
        print("⚠️ No valid experiences collected, skipping episode")
        continue
    
    print(f"✅ Collected {len(experiences)} experiences")
    
    # Step 2: Compute advantages
    rewards = [exp['reward'] for exp in experiences]
    values = [exp['value'] for exp in experiences]
    advantages, returns = compute_advantages(rewards, values)
    
    avg_reward = sum(rewards) / len(rewards)
    print(f"📊 Average reward: {avg_reward:.4f}")
    
    # Step 3: PPO Updates
    print("🔄 Performing PPO updates...")
    
    old_log_probs = torch.tensor([exp['log_prob'] for exp in experiences], device=DEVICE)
    advantages_tensor = torch.tensor(advantages, device=DEVICE)
    returns_tensor = torch.tensor(returns, device=DEVICE)
    
    for ppo_epoch in range(PPO_EPOCHS):
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for i, exp in enumerate(experiences):
            # Recompute log prob and value (with gradients for training)
            current_log_prob, current_value, entropy = compute_log_prob_and_value(exp['prompt'], exp['response'], requires_grad=True)
            
            # PPO policy loss
            ratio = torch.exp(current_log_prob - old_log_probs[i])
            surr1 = ratio * advantages_tensor[i]
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages_tensor[i]
            policy_loss = -torch.min(surr1, surr2)
            
            # Value loss
            value_loss = F.mse_loss(current_value, returns_tensor[i])
            
            # Entropy loss (negative because we want to maximize entropy)
            entropy_loss = -entropy
            
            # Total loss
            total_loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss
            
            # Backward pass
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy_model.parameters() if p.requires_grad],
                MAX_GRAD_NORM
            )
            torch.nn.utils.clip_grad_norm_(value_head.parameters(), MAX_GRAD_NORM)
            
            # Update
            policy_optimizer.step()
            value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        if ppo_epoch == 0:  # Print metrics only once per episode
            print(f"   Policy Loss: {total_policy_loss/len(experiences):.4f}")
            print(f"   Value Loss: {total_value_loss/len(experiences):.4f}")
            print(f"   Entropy Loss: {total_entropy_loss/len(experiences):.4f}")
    
    # Step 4: Save checkpoint every 5 episodes
    if (episode + 1) % 5 == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"simple_ppo_lora_ep{episode+1}_{timestamp}.pt"
        save_lora_weights(policy_model, checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # Step 5: Show sample results
    if episode % 5 == 0:
        print("\n🎯 Sample Results:")
        sample_prompt = batch_prompts[0]
        sample_response, _ = generate_response(sample_prompt)
        sample_reward = get_reward(sample_prompt, sample_response)
        
        print(f"   Prompt: {sample_prompt[:100]}...")
        print(f"   Response: {sample_response}")
        print(f"   Reward: {sample_reward:.4f}")

# 6. Final save
print("\n💾 Saving final model...")
final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_path = f"../simple_ppo_lora_final_{final_timestamp}.pt"
save_lora_weights(policy_model, final_path)

# Save value head too
value_path = f"../simple_ppo_value_final_{final_timestamp}.pt"
torch.save(value_head.state_dict(), value_path)

print(f"✅ Training completed!")
print(f"   Final LoRA weights: {final_path}")
print(f"   Value head weights: {value_path}")

# 7. Quick test
print("\n🧪 Quick test of trained model...")
test_prompt = "AI is transforming healthcare and business worldwide"
test_response, _ = generate_response(test_prompt)
test_reward = get_reward(test_prompt, test_response)

print(f"Test prompt: {test_prompt}")
print(f"Test response: {test_response}")
print(f"Test reward: {test_reward:.4f}") 