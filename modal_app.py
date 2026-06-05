# pyrefly: ignore [missing-import]
import modal
import os
import subprocess
import sys
import shutil

app = modal.App("optimus-continual-training")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "colorama",
        "psutil"
    )
    .add_local_file("config.py", remote_path="/app/config.py")
    .add_local_file("model.py", remote_path="/app/model.py")
    .add_local_file("dataset.py", remote_path="/app/dataset.py")
    .add_local_file("scheduler.py", remote_path="/app/scheduler.py")
    .add_local_file("train.py", remote_path="/app/train.py")
    .add_local_file("generate.py", remote_path="/app/generate.py")
    .add_local_file("download_data.py", remote_path="/app/download_data.py")
    .add_local_file("continual_trainer.py", remote_path="/app/continual_trainer.py")
)

vol = modal.Volume.from_name("optimus-data-vol", create_if_missing=True)

@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": vol},
    timeout=86400, # Max run time: 24 hours
)
def run_training():
    os.chdir("/app")
    
    # Create required directories on the persistent volume
    os.makedirs("/data/modal_checkpoints", exist_ok=True)
    os.makedirs("/data/data", exist_ok=True)
    
    local_checkpoint = "/app/checkpoints/latest_checkpoint.pt"
    modal_checkpoint = "/data/modal_checkpoints/latest_checkpoint.pt"
    uploaded_checkpoint = "/data/latest_checkpoint.pt"
    
    # If the user used `modal volume put` to manually upload the massive checkpoint directly to the volume root:
    if os.path.exists(uploaded_checkpoint) and not os.path.exists(modal_checkpoint):
        print("📥 Moving manually uploaded checkpoint into the modal_checkpoints directory...")
        shutil.move(uploaded_checkpoint, modal_checkpoint)
    elif os.path.exists(local_checkpoint) and not os.path.exists(modal_checkpoint):
        # Fallback if somehow it's mounted
        shutil.copy2(local_checkpoint, modal_checkpoint)
        
    local_state = "/app/continual_state.json"
    modal_state = "/data/continual_state.json"
    if os.path.exists(local_state) and not os.path.exists(modal_state):
        print("📥 Copying local continual state to Modal persistent volume...")
        shutil.copy2(local_state, modal_state)

    # Set up environment variables to point our local scripts to the persistent cloud volume
    env = os.environ.copy()
    env["OPTIMUS_CHECKPOINT_DIR"] = "/data/modal_checkpoints"
    env["OPTIMUS_STATE_FILE"] = "/data/continual_state.json"
    env["OPTIMUS_DATA_FILE"] = "/data/data/fineweb_chunk.txt"
    env["OPTIMUS_MODAL"] = "1"
    
    print("🚀 Starting continual trainer on Modal (T4)...")
    subprocess.run([sys.executable, "-u", "continual_trainer.py"], env=env)

@app.local_entrypoint()
def main():
    print("Initiating Modal Continual Trainer...")
    run_training.remote()

