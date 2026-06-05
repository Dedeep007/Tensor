import os
import sys
import json
import time
import subprocess
from datasets import load_dataset
from itertools import islice

# You must have the same tokenizer available to pre-filter lengths
from dataset import get_tokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PYTHON_EXE = sys.executable
STATE_FILE = "finetune_state.json"
CHECKPOINT_DIR = os.path.join(os.getcwd(), "chkpoints_finetune")

SAMPLES_PER_CHUNK = 20_000   # How many VALID examples to pull per chunk
MAX_SEQ_LENGTH = 1024        # Drop anything longer than this
TRAIN_SPLIT_RATIO = 0.9      # 90% train, 10% val

# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"skip_n": 0, "chunk_count": 0}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

# ---------------------------------------------------------------------------
# Command Runner
# ---------------------------------------------------------------------------
def run_command(cmd, desc):
    print(f"\n{'-'*60}")
    print(f"🔄 Running: {desc}")
    print(f"{'-'*60}\n")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    for line in process.stdout:
        print(line, end='', flush=True)
    process.wait()
    return process.returncode

# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    print("🤖 OPTIMUS Automated Continual Fine-Tuner Started!")
    print(f"Dataset: HuggingFaceTB/smoltalk")
    print("Press Ctrl+C at any time to safely stop.\n")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    tokenizer = get_tokenizer()
    
    while True:
        state = load_state()
        skip_n = state["skip_n"]
        chunk_num = state["chunk_count"] + 1
        print(f"\n🚀 --- STARTING FINETUNE CHUNK {chunk_num} ---")

        DATA_FILE_TEMP = f"finetune_chunk_{chunk_num}_train.jsonl"
        VAL_FILE_TEMP = f"finetune_chunk_{chunk_num}_val.jsonl"
        
        # We need to know how many raw items we skipped to reach this chunk
        raw_items_processed = skip_n

        if os.path.exists(DATA_FILE_TEMP) and os.path.exists(VAL_FILE_TEMP):
            print(f"✅ Found existing data files for chunk {chunk_num}, skipping download.")
            # If we resume from cache, we must trust the state for the next step
        else:
            print(f"📡 Streaming HuggingFaceTB/smoltalk...")
            full_dataset = load_dataset(
                "HuggingFaceTB/smoltalk",
                "all",
                split="train",
                streaming=True
            )
            
            iterator = iter(full_dataset)
            
            # Fast-forward past already processed RAW items
            print(f"⏭️  Fast-forwarding {raw_items_processed} items...")
            for _ in range(raw_items_processed):
                next(iterator, None)
                
            valid_items = []
            dropped_items = 0
            
            print(f"🔍 Filtering sequences to fit strictly within {MAX_SEQ_LENGTH} tokens...")
            # We pull items until we have exactly SAMPLES_PER_CHUNK valid ones
            while len(valid_items) < SAMPLES_PER_CHUNK:
                try:
                    item = next(iterator)
                    raw_items_processed += 1
                except StopIteration:
                    print("✅ Reached the end of the smoltalk dataset!")
                    break
                    
                # Format using ChatML template
                try:
                    # smoltalk has 'messages' list
                    messages = item.get("messages", [])
                    if not messages:
                        continue
                        
                    # Apply template
                    formatted_text = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
                    
                    # Check token length
                    tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
                    if len(tokens) <= MAX_SEQ_LENGTH:
                        # Keep it! We save the formatted text to avoid re-formatting in train.py
                        valid_items.append({"text": formatted_text})
                    else:
                        dropped_items += 1
                        
                except Exception as e:
                    dropped_items += 1
                    
            if not valid_items:
                print("❌ No valid items found. Exiting.")
                break
                
            print(f"✅ Found {len(valid_items)} valid conversations. Dropped {dropped_items} overly long conversations.")
            
            # Split downloaded items into train and val
            split_idx = int(len(valid_items) * TRAIN_SPLIT_RATIO)
            train_items = valid_items[:split_idx]
            val_items = valid_items[split_idx:]
            
            print("💾 Saving temporary chunk files...")
            with open(DATA_FILE_TEMP, "w", encoding="utf-8") as f:
                for item in train_items:
                    f.write(json.dumps(item, default=str) + "\n")
                    
            with open(VAL_FILE_TEMP, "w", encoding="utf-8") as f:
                for item in val_items:
                    f.write(json.dumps(item, default=str) + "\n")
                    
        # Train on the new chunk
        train_success = False
        while not train_success:
            train_cmd = [
                PYTHON_EXE, "-u", "finetune.py", 
                "--data_file", DATA_FILE_TEMP, 
                "--val_file", VAL_FILE_TEMP, 
                "--ckpt_dir", CHECKPOINT_DIR
            ]
            retcode = run_command(train_cmd, f"Fine-Tuning on Chunk {chunk_num}")
            if retcode == 0:
                train_success = True
            else:
                print(f"❌ Training crashed with exit code {retcode}. Retrying in 10 seconds...")
                time.sleep(10)

        # Update state for next chunk
        state["skip_n"] = raw_items_processed
        state["chunk_count"] = chunk_num
        save_state(state)
        
        # Clean up chunk files to save space
        if os.path.exists(DATA_FILE_TEMP): os.remove(DATA_FILE_TEMP)
        if os.path.exists(VAL_FILE_TEMP): os.remove(VAL_FILE_TEMP)
        
        print(f"🎉 Chunk {chunk_num} complete. State saved.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Orchestrator stopped by user. Goodbye!")
        sys.exit(0)
