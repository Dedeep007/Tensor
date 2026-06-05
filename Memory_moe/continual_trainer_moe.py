"""
OPTIMUS_moe Continual Trainer
===============================
Automated continual learning pipeline for the Memory MoE model.

Orchestrates:
    1. Download 1 lakh (100K) FineWeb-Edu samples per chunk
    2. Evaluate generation quality (pre-training)
    3. Train MoE on the chunk (frozen base + trainable MoE)
    4. Slide the data window and repeat

All data, checkpoints, and logs are stored in the Memory_moe/ folder.

Usage:
    conda activate EPOCH
    cd Memory_moe
    python continual_trainer_moe.py
"""

import os
import sys
import subprocess
import json
import time

# Fix Windows console encoding for emoji logging
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class Logger(object):
    """Dual logger: writes to both terminal and log file."""
    def __init__(self, filename="training_pipeline_moe.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger()
sys.stderr = sys.stdout

STATE_FILE = "continual_state_moe.json"
DATA_FILE = os.path.join("data", "fineweb_chunk.txt")
SAMPLES_PER_CHUNK = 100_000  # 1 lakh
PYTHON_EXE = sys.executable


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"skip_n": 0, "chunk_count": 0}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def run_command(cmd, desc):
    print(f"\n{'='*60}")
    print(f"🔄 {desc}")
    print(f"{'='*60}\n")
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace'
    )
    
    for line in process.stdout:
        print(line, end='', flush=True)
        
    process.wait()
    return process.returncode


def main():
    # Mark that continual trainer is running to prevent duplicate logging in train_moe.py
    os.environ["CONTINUAL_TRAINER_ACTIVE"] = "1"

    print("🤖 OPTIMUS_moe Automated Continual Trainer Started!")
    print("🧠 Phase 1: Frozen Base + Trainable Memory MoE")
    print("Press Ctrl+C at any time to safely stop.\n")
    
    state = load_state()
    
    while True:
        state = load_state()
            
        skip_n = state["skip_n"]
        chunk_num = state["chunk_count"] + 1
        
        print(f"\n🚀 --- STARTING CHUNK {chunk_num} ---")
        
        # 1. Check existing chunk or download new chunk
        download_success = False
        if os.path.exists(DATA_FILE):
            if os.path.getsize(DATA_FILE) > 1000:
                print(f"✅ Found existing data chunk ({DATA_FILE}), skipping download...")
                download_success = True
            else:
                os.remove(DATA_FILE)
                
        while not download_success:
            cmd = [
                PYTHON_EXE, "-u", "download_data_moe.py",
                "--num_samples", str(SAMPLES_PER_CHUNK),
                "--skip_n", str(skip_n),
                "--output", DATA_FILE,
            ]
            retcode = run_command(cmd, f"Downloading {SAMPLES_PER_CHUNK} samples (skip={skip_n})")
            
            if retcode == 0 and os.path.exists(DATA_FILE):
                download_success = True
            else:
                print("⚠️ Download failed or network timed out. Retrying in 10 seconds...")
                time.sleep(10)
        
        # 2. Train MoE on the chunk
        train_success = False
        while not train_success:
            train_cmd = [PYTHON_EXE, "-u", "train_moe.py"]
            retcode = run_command(train_cmd, f"Training MoE on Chunk {chunk_num}")
            
            if retcode == 0:
                train_success = True
            else:
                print(f"❌ Training crashed with exit code {retcode}. Check logs above.")
                print("Retrying training in 10 seconds...")
                time.sleep(10)
        
        # 3. Update state for next chunk (20% sliding window)
        state["skip_n"] += int(SAMPLES_PER_CHUNK * 0.2)
        state["chunk_count"] += 1
        save_state(state)
        
        # 4. Clean up old chunk
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
            
        print(f"\n✅ Chunk {chunk_num} complete. State saved. Moving to next chunk in 5 seconds...\n")
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Continual trainer stopped by user. State is saved safely!")
