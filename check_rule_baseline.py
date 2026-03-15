import torch, numpy as np
from pathlib import Path

data_dir = Path("/Users/seb/Desktop/bcbl/msc_thesis/code/gm_data_erb_1300_1700")
X = torch.load(data_dir/"input_blocks.pt").numpy().astype(np.float32)  # (B,10,8)
Y = torch.load(data_dir/"labels_blocks.pt").numpy().astype(np.int64)   # (B,10) values 4/5/6

# rule: assume first 3 tones are "standard", deviant is the tone (among 4/5/6) farthest from std mean
# (this matches the common oddball design: standard repeated, deviant different)
std_mean = X[:,:,0:3].mean(axis=2)  # (B,10)
candidates = X[:,:,3:6]             # positions 4,5,6 -> indices 3,4,5
dist = np.abs(candidates - std_mean[:,:,None])  # (B,10,3)
pred = dist.argmax(axis=2) + 4      # back to label space {4,5,6}

acc = (pred == Y).mean()
print("Rule baseline acc:", acc)

# extra diagnostics: how often first 3 are exactly equal (or near equal)
eq = np.isclose(X[:,:,0], X[:,:,1]) & np.isclose(X[:,:,1], X[:,:,2])
print("First-3 equal ratio:", eq.mean())

# deviant strength: average dist of true label vs non-true
true_idx = (Y - 4)  # 0/1/2
true_dist = dist[np.arange(dist.shape[0])[:,None], np.arange(dist.shape[1])[None,:], true_idx]
other_dist = dist.mean(axis=2)
print("Mean true_dist:", true_dist.mean(), "Mean dist(avg over 4-6):", other_dist.mean())
