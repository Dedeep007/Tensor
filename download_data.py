import os
import sys
import argparse
import time
from datasets import load_dataset

# Fix Windows console encoding for emoji logging
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def main():
    parser = argparse.ArgumentParser(description="Download a chunk of FineWeb-Edu dataset locally.")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of documents to download")
    parser.add_argument("--skip_n", type=int, default=0, help="Number of documents to skip (for fetching new chunks)")
    parser.add_argument("--output", type=str, default="data/chunk_1.txt", help="Output text file path")
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"🌐 Connecting to HuggingFace FineWeb-Edu...")
    print(f"   Target samples: {args.num_samples}")
    print(f"   Skipping:       {args.skip_n}")
    
    try:
        # Load the dataset in streaming mode
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2013-20", split="train", streaming=True)
        
        if args.skip_n > 0:
            print(f"⏭️  Skipping {args.skip_n} documents (this happens internally and takes a few minutes silently)...")
            ds = ds.skip(args.skip_n)
            
        stream = iter(ds)
        
        print(f"📥 Downloading to {args.output}...")
        start_time = time.time()
        
        with open(args.output, "w", encoding="utf-8") as f:
            count = 0
            while count < args.num_samples:
                try:
                    # Get next document
                    doc = next(stream)
                    text = doc.get("text", "").strip()
                    
                    if len(text) > 50:  # Only save meaningful paragraphs
                        f.write(text + "\n\n")
                        count += 1
                        
                        if count % 100 == 0:
                            elapsed = time.time() - start_time
                            print(f"   Progress: {count}/{args.num_samples} ({count/elapsed:.1f} docs/sec)")
                            
                except StopIteration:
                    print("\n⚠️ Stream exhausted!")
                    break
                except Exception as e:
                    print(f"\n⚠️ Network error while fetching document {count}: {e}")
                    print("   Retrying in 5 seconds...")
                    time.sleep(5)
                    # Re-initialize stream if it breaks completely
                    stream = iter(ds.skip(args.skip_n + count))
                    
        print(f"\n✅ Successfully saved {count} documents to {args.output}")
        # Force immediate exit to prevent HuggingFace datasets background threads from crashing during teardown
        os._exit(0)
        
    except Exception as e:
        print(f"❌ Failed to connect to HuggingFace datasets: {e}")

if __name__ == "__main__":
    main()
