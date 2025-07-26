
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

## works as envirement.
examples = [
    ("this location is good and", "beautiful"),
    ("the dog likes to", "run"),
    ("he went to the", "market"),
    ("i am feeling very", "happy"),
]

words = set()
for p, t in examples:
    for w in p.split(): words.add(w)
    words.add(t)
    
words.add("<unk>")
    
vocab = sorted(words)
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}
vocab_size = len(vocab)

prefix_token_idxs = []
target_idxs = []
for p, t in examples:
    prefix_idxs = [word_to_idx[w] for w in p.split()]
    prefix_token_idxs.append(prefix_idxs)
    target_idxs.append(word_to_idx[t])

max_prefix_len = max(len(x) for x in prefix_token_idxs)

PAD = vocab_size
for i in range(len(prefix_token_idxs)):
    pad_len = max_prefix_len - len(prefix_token_idxs[i])
    prefix_token_idxs[i] = prefix_token_idxs[i] + [PAD] * pad_len

## simple model??
class TinyLM(nn.Module):
    def __init__(self, vocab_size, pad_idx, emb=32, hid=32):
        super().__init__()
    
        self.embed = nn.Embedding(vocab_size + 1, emb, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.lstm1 = nn.LSTM(hid, hid, batch_first=True)
        self.lstm2 = nn.LSTM(hid, hid, batch_first=True)
        self.fc = nn.Linear(hid, vocab_size) 
    def forward(self, x):
    
        e = self.embed(x)              
        out, (h, c) = self.lstm1(e) 
        out, (h, c) = self.lstm2(out)
        out, (h, c) = self.lstm2(out)                  
        last = out[:, -1, :]           
        logits = self.fc(last)         
        return logits

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = TinyLM(vocab_size=vocab_size, pad_idx=PAD).to(device)
opt = optim.AdamW(model.parameters(), lr=1e-3)

prefixes_tensor = torch.tensor(prefix_token_idxs, dtype=torch.long, device=device) 
targets_tensor = torch.tensor(target_idxs, dtype=torch.long, device=device)        

epochs = 800
batch_size = len(examples) 

for ep in range(1, epochs + 1):
    model.train()
    opt.zero_grad()
    logits = model(prefixes_tensor)          
    probs = torch.softmax(logits, dim=-1)    
    dists = torch.distributions.Categorical(probs)
    sampled = dists.sample()                 
    rewards = (sampled == targets_tensor).float() 


    logp = dists.log_prob(sampled)
    loss = -(rewards * logp).mean()

    loss.backward()
    opt.step()

    if ep % 100 == 0 or ep == 1:
        avg_r = rewards.mean().item()
        print(f"Ep {ep:4d} | loss {loss.item():.4f} | avg_sampled_reward {avg_r:.2f}")

model.eval()
with torch.no_grad():
    logits = model(prefixes_tensor)           
    greedy = torch.argmax(logits, dim=-1)     
    correct = (greedy == targets_tensor).long()
    for i, (prefix, _) in enumerate(examples):
        pred_word = idx_to_word[int(greedy[i].cpu())]
        target_word = idx_to_word[int(targets_tensor[i].cpu())]
        print(f"Prefix: '{prefix}'")
        print(f"  Target: '{target_word}' | Predicted: '{pred_word}'")
    acc = correct.float().mean().item()
    print(f"\nGreedy accuracy on prefixes: {acc:.2%}")

def predict_next_word(model, input_str):
    model.eval()
    with torch.no_grad():
        toks = input_str.split()
        toks_idx = [word_to_idx.get(w, word_to_idx["<unk>"]) for w in toks] + [PAD] * (max_prefix_len - len(toks))
        t = torch.tensor([toks_idx], dtype=torch.long, device=device)
        logits = model(t)
        pred_idx = torch.argmax(logits, dim=-1)[0].item()
        return idx_to_word[pred_idx]

user_input = input("Ask something:")
next_word = predict_next_word(model, user_input)
print(f"{user_input} {next_word}")
