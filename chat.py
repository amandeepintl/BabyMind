import os
import pickle
import torch
import tiktoken
from model import GPTConfig, GPT

# --- config ---
out_dir = 'out-shakespeare-char'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_new_tokens = 200
temperature = 0.8
top_k = 40

# --- load model ---
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# --- load vocab ---
meta_path = os.path.join('data', 'shakespeare_char', 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi = meta['stoi']
itos = meta['itos']

def encode(s):
    # only encode chars the model knows
    return [stoi[c] for c in s if c in stoi]

def decode(l):
    return ''.join([itos[i] for i in l])

print("=" * 50)
print("  BabyMind — Chat Mode")
print("  Your AI has learned from what you taught it.")
print("  Type anything and see what it continues.")
print("  Type 'quit' to exit.")
print("=" * 50)
print()

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if user_input.lower() in ('quit', 'exit', 'bye'):
        print("AI: shukriya! bye bye!")
        break

    if not user_input:
        continue

    # encode input
    encoded = encode(user_input)
    if not encoded:
        print("AI: ...? (unknown characters, try Hindi/English letters only)")
        continue

    x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

    # decode and strip the input prefix
    full_output = decode(y[0].tolist())
    response = full_output[len(user_input):]

    # clean up — take only first few lines
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    short_response = '\n'.join(lines[:4]) if lines else '...'

    print(f"AI: {short_response}")
    print()
