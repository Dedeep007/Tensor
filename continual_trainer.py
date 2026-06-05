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
    def __init__(self, filename="training_pipeline.log"):
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

STATE_FILE = os.environ.get("OPTIMUS_STATE_FILE", "continual_state.json")
DATA_FILE = os.environ.get("OPTIMUS_DATA_FILE", "data/fineweb_chunk.txt")
SAMPLES_PER_CHUNK = 50000
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
    
    # Run process and pipe output to stdout in real-time
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    
    for line in process.stdout:
        print(line, end='', flush=True)
        
    process.wait()
    return process.returncode

def main():
    print("🤖 OPTIMUS Automated Continual Trainer Started!")
    print("Press Ctrl+C at any time to safely stop. The state is saved automatically.\n")
    
    state = load_state()
    
    while True:
        state = load_state()
            
        skip_n = state["skip_n"]
        chunk_num = state["chunk_count"] + 1
        
        print(f"\n🚀 --- STARTING CHUNK {chunk_num} ---")
        
        # 1 & 2. Check existing chunk or download new chunk
        download_success = False
        if os.path.exists(DATA_FILE):
            # Check if file has some size
            if os.path.getsize(DATA_FILE) > 1000:
                print(f"✅ Found existing data chunk ({DATA_FILE}), skipping download...")
                download_success = True
            else:
                os.remove(DATA_FILE)
                
        # Load dataset chunk using HuggingFace datasets library
        from datasets import load_dataset
        # Load the full dataset (streaming disabled for slicing)
        full_dataset = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train")
        total_size = len(full_dataset)
        # Determine start and end indices for the current chunk
        start_idx = skip_n
        end_idx = min(skip_n + SAMPLES_PER_CHUNK, total_size)
        if start_idx >= total_size:
            print("✅ No more data to process. Exiting.")
            break
        chunk = full_dataset.select(range(start_idx, end_idx))
        # Save chunk to a temporary file for downstream scripts (if needed)
        DATA_FILE_TEMP = f"temp_chunk_{chunk_num}.jsonl"
        chunk.to_json(DATA_FILE_TEMP, orient="records", lines=True)
        download_success = True
        print(f"✅ Loaded dataset chunk {chunk_num}: records {start_idx} to {end_idx - 1}")
                
        # 3. Evaluate generation quality BEFORE training on chunk
        print("\n📝 Evaluating Grammar BEFORE training this chunk...")
        eval_cmd = [PYTHON_EXE, "-u", "generate.py", "--compare", "--prompt", "The future of AI is", "--max_tokens", "150", "--temperature", "1.0"]
        run_command(eval_cmd, f"Evaluating Grammar for Chunk {chunk_num} (Pre-Train)")
        
        # 4. Train on new chunk
        train_success = False
        while not train_success:
            train_cmd = [PYTHON_EXE, "-u", "train.py"]
            retcode = run_command(train_cmd, f"Training on Chunk {chunk_num}")
            
            if retcode == 0:
                train_success = True
            else:
                print(f"❌ Training crashed with exit code {retcode}. Check logs above.")
                print("Retrying training in 10 seconds...")
                time.sleep(10)
        
        # 5. Update state for next chunk with a sliding window (80% overlap)
        state["skip_n"] = end_idx - int(SAMPLES_PER_CHUNK * 0.2)  # overlap 20%
        state["chunk_count"] += 1
        save_state(state)
        
        # 6. Clean up temporary chunk file
        if os.path.exists(DATA_FILE_TEMP):
            os.remove(DATA_FILE_TEMP)
            
        print(f"\n✅ Chunk {chunk_num} complete. State saved. Temp data deleted. Moving to next chunk in 5 seconds...\n")
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Continual trainer stopped by user. State is saved safely!")
