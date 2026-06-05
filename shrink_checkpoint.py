import torch
import os

filepath = "checkpoints/latest_checkpoint.pt"
outpath = "checkpoints/light_checkpoint.pt"

print(f"Loading {filepath}...")
ckpt = torch.load(filepath, map_location="cpu")

print("Original keys:", ckpt.keys())

# Strip heavy optimizer states
for key in ["optimizer_state_dict", "scaler_state_dict"]:
    if key in ckpt:
        print(f"Removing {key}...")
        del ckpt[key]

print(f"Saving stripped checkpoint to {outpath}...")
torch.save(ckpt, outpath)

old_size = os.path.getsize(filepath) / (1024**3)
new_size = os.path.getsize(outpath) / (1024**3)

print(f"✅ Shrink complete!")
print(f"   Original size: {old_size:.2f} GB")
print(f"   New size:      {new_size:.2f} GB")
