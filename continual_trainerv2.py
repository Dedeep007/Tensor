import os
import sys
import subprocess
import json
import time

# Load unified configuration
from configv2 import TrainingConfigV2
cfg = TrainingConfigV2()

# Fix Windows console encoding for emoji logging
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

class Logger(object):
    def __init__(self, filename=None):
        # If no filename provided, default to a log file inside the checkpoint directory
        if filename is None:
            filename = os.path.join(cfg.checkpoint_dir, "training.log")
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

# Environment/constant configuration
DATASET_SUBSET = "cosmopedia-v2" # Change this back to "fineweb-edu-dedup" whenever you want to resume fineweb!

if DATASET_SUBSET == "fineweb-edu-dedup":
    STATE_FILE = os.environ.get("OPTIMUS_STATE_FILE", "continual_state_v2.json") # Keep old filename for fineweb to save progress
else:
    STATE_FILE = os.environ.get("OPTIMUS_STATE_FILE", f"{DATASET_SUBSET}_state_v2.json")

SAMPLES_PER_CHUNK = cfg.samples_per_chunk  # use config
PYTHON_EXE = sys.executable
# Directory to store checkpoints and logs (separate from code)
CHECKPOINT_DIR = cfg.checkpoint_dir
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_state(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {"skip_n": 0, "chunk_count": 0}

def save_state(state, filepath):
    with open(filepath, "w") as f:
        json.dump(state, f)

def run_command(cmd, desc):
    print(f"\n{'='*60}")
    print(f"🔄 {desc}")
    print(f"{'='*60}\n")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        process.wait()
    except KeyboardInterrupt:
        print("\n[Orchestrator] Caught Ctrl+C! Waiting for trainer to save checkpoint and exit gracefully...")
        process.wait()  # Block and give train.py time to write the massive .pt file
        raise  # Re-raise to cleanly exit the orchestrator
    return process.returncode

def main():
    print("🤖 OPTIMUS Automated Continual Trainer v2 Started!")
    print("Press Ctrl+C at any time to safely stop. The state is saved automatically.\n")
    while True:
        if DATASET_SUBSET == "hybrid":
            fw_state_file = os.environ.get("OPTIMUS_STATE_FILE", "continual_state_v2.json")
            cosmo_state_file = os.environ.get("OPTIMUS_STATE_FILE", "cosmopedia-v2_state_v2.json")
            py_state_file = os.environ.get("OPTIMUS_STATE_FILE", "python-edu_state_v2.json")
            
            fw_state = load_state(fw_state_file)
            cosmo_state = load_state(cosmo_state_file)
            py_state = load_state(py_state_file)
            
            chunk_num = fw_state["chunk_count"] + 1
            print(f"\n🚀 --- STARTING CHUNK {chunk_num} (HYBRID MIX) ---")
            
            DATA_FILE_TEMP = f"temp_hybrid_chunk_{chunk_num}_train.jsonl"
            VAL_FILE_TEMP = f"temp_hybrid_chunk_{chunk_num}_val.jsonl"
            
            samples_per_subset = SAMPLES_PER_CHUNK // 3
            
            if os.path.exists(DATA_FILE_TEMP) and os.path.exists(VAL_FILE_TEMP):
                print(f"✅ Found existing hybrid data files for chunk {chunk_num}, skipping download.")
            else:
                from datasets import load_dataset
                from itertools import islice
                import random
                
                def fetch_subset(subset_name, skip_n, count):
                    while True:
                        try:
                            print(f"Fetching {count} records from {subset_name} (skipping {skip_n})...")
                            ds = load_dataset("HuggingFaceTB/smollm-corpus", subset_name, split="train", streaming=True)
                            iterator = iter(ds)
                            for _ in range(skip_n):
                                next(iterator, None)
                            return list(islice(iterator, count))
                        except Exception as e:
                            print(f"❌ Network error fetching {subset_name}: {e}. Retrying in 10 seconds...")
                            time.sleep(10)
                
                fw_items = fetch_subset("fineweb-edu-dedup", fw_state["skip_n"], samples_per_subset)
                cosmo_items = fetch_subset("cosmopedia-v2", cosmo_state["skip_n"], samples_per_subset)
                py_items = fetch_subset("python-edu", py_state["skip_n"], samples_per_subset)
                
                chunk_items = fw_items + cosmo_items + py_items
                random.shuffle(chunk_items)
                
                train_split_ratio = cfg.train_split
                split_idx = int(len(chunk_items) * train_split_ratio)
                train_items = chunk_items[:split_idx]
                val_items = chunk_items[split_idx:]
                
                with open(DATA_FILE_TEMP, "w", encoding="utf-8") as f:
                    for item in train_items:
                        f.write(json.dumps(item, default=str) + "\n")
                with open(VAL_FILE_TEMP, "w", encoding="utf-8") as f:
                    for item in val_items:
                        f.write(json.dumps(item, default=str) + "\n")
                
                print(f"✅ Loaded hybrid chunk {chunk_num}: ({len(train_items)} train, {len(val_items)} val)")
                
        else:
            state = load_state(STATE_FILE)
            skip_n = state["skip_n"]
            chunk_num = state["chunk_count"] + 1
            print(f"\n🚀 --- STARTING CHUNK {chunk_num} ({DATASET_SUBSET}) ---")

            if DATASET_SUBSET == "fineweb-edu-dedup":
                DATA_FILE_TEMP = f"temp_chunk_{chunk_num}_train.jsonl"
                VAL_FILE_TEMP = f"temp_chunk_{chunk_num}_val.jsonl"
            else:
                DATA_FILE_TEMP = f"temp_{DATASET_SUBSET}_chunk_{chunk_num}_train.jsonl"
                VAL_FILE_TEMP = f"temp_{DATASET_SUBSET}_chunk_{chunk_num}_val.jsonl"
            start_idx = skip_n

            if os.path.exists(DATA_FILE_TEMP) and os.path.exists(VAL_FILE_TEMP):
                print(f"✅ Found existing {DATASET_SUBSET} data files for chunk {chunk_num}, skipping download.")
                end_idx = start_idx + SAMPLES_PER_CHUNK
            else:
                # Load a chunk from HuggingFace dataset using streaming mode
                from datasets import load_dataset
                from itertools import islice
                
                download_success = False
                while not download_success:
                    try:
                        full_dataset = load_dataset(
                            "HuggingFaceTB/smollm-corpus",
                            DATASET_SUBSET,
                            split="train",
                            streaming=True  # enable streaming to fetch only needed samples
                        )
                        # Total number of examples is available via .info
                        total_size = full_dataset.info.splits["train"].num_examples
                        if start_idx >= total_size:
                            print("✅ No more data to process. Exiting.")
                            sys.exit(0)
                            
                        # Advance iterator to start_idx
                        iterator = iter(full_dataset)
                        for _ in range(start_idx):
                            next(iterator, None)
                            
                        # Take a slice of SAMPLES_PER_CHUNK items
                        chunk_items = list(islice(iterator, SAMPLES_PER_CHUNK))
                        
                        # Split downloaded items into train and val
                        train_split_ratio = cfg.train_split
                        split_idx = int(len(chunk_items) * train_split_ratio)
                        train_items = chunk_items[:split_idx]
                        val_items = chunk_items[split_idx:]
                        download_success = True
                    except Exception as e:
                        print(f"❌ Network error during dataset fetch: {e}. Retrying in 10 seconds...")
                        time.sleep(10)
                
                with open(DATA_FILE_TEMP, "w", encoding="utf-8") as f:
                    for item in train_items:
                        f.write(json.dumps(item, default=str) + "\n")
                        
                with open(VAL_FILE_TEMP, "w", encoding="utf-8") as f:
                    for item in val_items:
                        f.write(json.dumps(item, default=str) + "\n")
                        
                end_idx = start_idx + len(chunk_items)
                print(f"✅ Loaded {DATASET_SUBSET} chunk {chunk_num}: records {start_idx} to {end_idx - 1} ({len(train_items)} train, {len(val_items)} val)")

        # 4. Train on new chunk
        train_success = False
        while not train_success:
            # Pass both data files to train.py
            train_cmd = [PYTHON_EXE, "-u", "train.py", "--data_file", DATA_FILE_TEMP, "--val_file", VAL_FILE_TEMP, "--ckpt_dir", CHECKPOINT_DIR]
            desc_name = "HYBRID MIX" if DATASET_SUBSET == "hybrid" else DATASET_SUBSET
            retcode = run_command(train_cmd, f"Training on {desc_name} Chunk {chunk_num}")
            if retcode == 0:
                train_success = True
            else:
                print(f"❌ Training crashed with exit code {retcode}. Retrying in 10 seconds...")
                time.sleep(10)

        # 5. Update state for next chunk (no overlap, 100% new data)
        if DATASET_SUBSET == "hybrid":
            fw_state["skip_n"] += samples_per_subset
            cosmo_state["skip_n"] += samples_per_subset
            py_state["skip_n"] += samples_per_subset
            
            fw_state["chunk_count"] += 1
            cosmo_state["chunk_count"] += 1
            py_state["chunk_count"] += 1
            
            save_state(fw_state, fw_state_file)
            save_state(cosmo_state, cosmo_state_file)
            save_state(py_state, py_state_file)
        else:
            state["skip_n"] = end_idx
            state["chunk_count"] += 1
            save_state(state, STATE_FILE)

        # 6. Clean up temporary chunk files
        if os.path.exists(DATA_FILE_TEMP):
            os.remove(DATA_FILE_TEMP)
        if os.path.exists(VAL_FILE_TEMP):
            os.remove(VAL_FILE_TEMP)
        desc_name = "HYBRID MIX" if DATASET_SUBSET == "hybrid" else DATASET_SUBSET
        print(f"\n✅ {desc_name} Chunk {chunk_num} complete. State saved. Temp data deleted. Moving to next chunk in 5 seconds...\n")
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Continual trainer stopped by user. State is saved safely!")
