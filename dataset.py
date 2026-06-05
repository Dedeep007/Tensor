"""
OPTIMUS Dataset & Data Loading
===============================
Streaming Causal Language Modeling dataset using HuggingFace FineWeb
(HuggingFaceFW/fineweb, sample-10BT subset) and the Qwen2.5 tokenizer.

Design:
    - Streams FineWeb via HuggingFace `datasets` library (no full download)
    - Packs tokenized documents into fixed-length sequences for max GPU utilization
    - Separates a small held-out validation buffer from the stream
    - Falls back to local .txt files or placeholder data if FineWeb is unavailable
"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer


# ===========================================================================
# Tokenizer
# ===========================================================================

def get_tokenizer():
    """Load and configure the Qwen2.5 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ===========================================================================
# FineWeb Streaming Dataset (Token-Packed)
# ===========================================================================

class FineWebStreamDataset(IterableDataset):
    """
    Streaming dataset that pulls documents from HuggingFace FineWeb-Edu,
    tokenizes them on-the-fly, and packs tokens into fixed-length
    sequences with no padding waste.

    FineWeb-Edu (HuggingFaceFW/fineweb-edu):
        A 1.3 trillion token subset of FineWeb, filtered using AI
        classifiers to retain only high-quality educational and factual
        content. Currently considered gold standard for LLM pretraining.

    Token packing:
        Documents are concatenated with an EOS separator into a rolling
        buffer. Once the buffer has enough tokens, a (max_length) chunk
        is yielded as one training sample. This maximizes GPU utilization
        since every token in every batch is a real training signal.

    Args:
        tokenizer: The Qwen tokenizer instance.
        dataset_name: HuggingFace dataset name (default "HuggingFaceFW/fineweb-edu").
        max_length: Sequence length per sample (default 512).
        split: Which split to use (default "train").
        subset: Dataset config/subset name (default "sample-10BT").
        skip_n: Number of documents to skip (for validation offset).
        max_samples: Maximum number of packed sequences to yield (None = infinite).
        seed: Random seed for shuffling the stream buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset_name="HuggingFaceFW/fineweb-edu",
        max_length=512,
        split="train",
        subset="CC-MAIN-2013-20",
        skip_n=0,
        max_samples=None,
        seed=42,
        buffer_size=10_000,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.split = split
        self.subset = subset
        self.skip_n = skip_n
        self.max_samples = max_samples
        self.seed = seed
        self.buffer_size = buffer_size
        self.eos_token_id = tokenizer.eos_token_id

    def _get_stream(self):
        """Create the HuggingFace streaming dataset iterator."""
        from datasets import load_dataset

        ds = load_dataset(
            self.dataset_name,
            name=self.subset,
            split=self.split,
            streaming=True,
        )
        # Shuffle with a buffer for randomness
        ds = ds.shuffle(seed=self.seed, buffer_size=self.buffer_size)

        # Skip documents (used to offset validation data)
        if self.skip_n > 0:
            ds = ds.skip(self.skip_n)

        return iter(ds)

    def __iter__(self):
        """
        Yield packed token sequences of exactly (max_length) tokens.

        Process:
            1. Pull documents from FineWeb stream
            2. Tokenize each document
            3. Append tokens + EOS to a rolling buffer
            4. When buffer >= max_length, yield a chunk and shift buffer
        """
        stream = self._get_stream()
        token_buffer = []
        samples_yielded = 0

        for doc in stream:
            # Tokenize the document text
            text = doc.get("text", "")
            if not text or len(text.strip()) < 20:
                continue

            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            # Append document tokens + EOS separator
            token_buffer.extend(token_ids)
            token_buffer.append(self.eos_token_id)

            # Yield complete chunks from the buffer
            while len(token_buffer) >= self.max_length:
                chunk = token_buffer[: self.max_length]
                token_buffer = token_buffer[self.max_length :]

                input_ids = torch.tensor(chunk, dtype=torch.long)
                # For packed sequences, all positions are valid (no padding)
                attention_mask = torch.ones(self.max_length, dtype=torch.long)
                # Labels = input_ids (model handles shifting internally)
                labels = input_ids.clone()

                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

                samples_yielded += 1
                if self.max_samples is not None and samples_yielded >= self.max_samples:
                    return


# ===========================================================================
# Validation Buffer Dataset (Map-Style, from FineWeb)
# ===========================================================================

class ValidationBufferDataset(Dataset):
    """
    Pre-fetches a fixed number of packed sequences from FineWeb-Edu for
    use as a consistent validation set across training.

    This is a map-style dataset that materializes data in memory so
    validation is fast and reproducible.
    """

    def __init__(self, tokenizer, dataset_name="HuggingFaceFW/fineweb-edu",
                 max_length=512, num_samples=64, subset="CC-MAIN-2013-20", seed=12345):
        super().__init__()
        self.samples = []

        print(f"   📥 Pre-fetching {num_samples} validation samples from FineWeb-Edu...")
        
        # Use a different seed and skip offset to avoid overlap with training
        val_stream = FineWebStreamDataset(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            max_length=max_length,
            subset=subset,
            skip_n=500_000,       # Skip past training data region
            max_samples=num_samples,
            seed=seed,
            buffer_size=1_000,
        )

        for sample in val_stream:
            self.samples.append(sample)

        print(f"   ✅ Loaded {len(self.samples)} validation samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ===========================================================================
# Fallback: Local Text Dataset
# ===========================================================================

PLACEHOLDER_TEXTS = [
    "The transformer architecture has revolutionized natural language processing. "
    "At its core, the self-attention mechanism allows every token in a sequence to "
    "attend to every other token, capturing long-range dependencies that recurrent "
    "networks struggled with. Modern variants like Grouped Query Attention reduce "
    "the memory overhead of the key-value cache while maintaining model quality.",

    "Rotary Position Embeddings encode positional information by rotating the query "
    "and key vectors in pairs of dimensions. This elegant formulation naturally "
    "captures relative position information and extends well to longer sequences "
    "than those seen during training. The rotation angles are determined by a "
    "geometric progression of frequencies, similar to sinusoidal embeddings.",

    "The SwiGLU activation function combines the Swish activation with a gating "
    "mechanism. Given an input x, SwiGLU computes Swish(xW_gate) * (xW_up) and "
    "then projects back down with W_down. This gated approach provides better "
    "gradient flow compared to standard ReLU or GELU activations and has become "
    "the default choice in state-of-the-art language models.",

    "Pre-normalization applies layer normalization before the attention and "
    "feed-forward sublayers rather than after. This architecture choice, combined "
    "with RMS normalization instead of standard LayerNorm, improves training "
    "stability significantly. RMSNorm removes the mean-centering step, reducing "
    "computational cost while maintaining performance.",

    "Mixed precision training leverages lower-precision floating point formats "
    "like FP16 or BF16 for most computations while keeping a master copy of "
    "weights in FP32. This approach nearly doubles training throughput on modern "
    "GPUs with Tensor Cores while maintaining model quality through careful "
    "loss scaling and selective FP32 computation for numerically sensitive ops.",

    "Gradient accumulation enables training with effectively large batch sizes "
    "on limited GPU memory. Instead of updating weights after every micro-batch, "
    "gradients are accumulated over multiple forward-backward passes before a "
    "single optimizer step. This technique is essential for training large "
    "language models on consumer hardware.",

    "The cosine learning rate schedule with linear warmup has become the standard "
    "for training transformer models. During the warmup phase, the learning rate "
    "increases linearly from zero to the peak value, allowing the model to "
    "stabilize before aggressive updates. The subsequent cosine decay gradually "
    "reduces the learning rate, enabling fine-grained convergence.",

    "Key-value caching is critical for efficient autoregressive generation. "
    "During inference, each new token only requires computing attention with "
    "its own query against all previous keys and values. By caching the key "
    "and value projections from previous steps, we avoid redundant computation "
    "and achieve linear-time generation per token.",

    "Early stopping monitors validation loss during training and halts the "
    "process when performance stops improving. This regularization technique "
    "prevents overfitting by selecting the model checkpoint with the best "
    "generalization performance rather than the one that memorized the "
    "training data most completely.",

    "Modern language models use subword tokenization algorithms like BPE or "
    "SentencePiece to handle open-vocabulary text. These tokenizers break rare "
    "words into common subword units while keeping frequent words intact. The "
    "Qwen tokenizer, based on tiktoken, achieves excellent compression ratios "
    "across multilingual text with a vocabulary of over 150,000 tokens.",
]


class TextChunkDataset(Dataset):
    """
    Map-style dataset from a list of text strings. Used as fallback
    when FineWeb is unavailable (offline mode).
    Tokenizes lazily to avoid massive RAM usage and startup delays.
    """

    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize lazily
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        # Mask padding positions with -100 in labels
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ===========================================================================
# Builder Functions
# ===========================================================================

def build_fineweb_dataloaders(tokenizer, training_config, model_config):
    """
    Build train (streaming) and validation (buffered) DataLoaders
    using HuggingFace FineWeb.

    Returns:
        train_loader: DataLoader wrapping FineWebStreamDataset (iterable)
        val_loader: DataLoader wrapping ValidationBufferDataset (map-style)
    """
    print("🌐 Setting up FineWeb-Edu dataset (streaming)...")
    print(f"   Dataset:        {training_config.fineweb_name}")
    print(f"   Subset:         {training_config.fineweb_subset}")
    print(f"   Seq length:     {training_config.max_seq_length}")
    print(f"   Batch size:     {training_config.batch_size}")
    print(f"   Val samples:    {training_config.val_samples}")

    # --- Training: Streaming ---
    train_dataset = FineWebStreamDataset(
        tokenizer=tokenizer,
        dataset_name=training_config.fineweb_name,
        max_length=training_config.max_seq_length,
        subset=training_config.fineweb_subset,
        seed=42,
        buffer_size=10_000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        prefetch_factor=4 if training_config.num_workers > 0 else None,
    )

    # --- Validation: Pre-fetched buffer ---
    val_dataset = ValidationBufferDataset(
        tokenizer=tokenizer,
        dataset_name=training_config.fineweb_name,
        max_length=training_config.max_seq_length,
        num_samples=training_config.val_samples,
        subset=training_config.fineweb_subset,
        seed=12345,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
    )

    return train_loader, val_loader


def build_fallback_dataloaders(tokenizer, training_config, model_config):
    """
    Fallback: Build dataloaders from local .txt files or placeholder data.
    Used when FineWeb is unavailable (no internet / offline).
    """
    from torch.utils.data import random_split

    data_file = os.environ.get("OPTIMUS_DATA_FILE", os.path.join(os.path.dirname(__file__), "data", "fineweb_chunk.txt"))
    data_dir = os.path.dirname(data_file)

    if os.path.isdir(data_dir):
        texts = _load_texts_from_directory(data_dir)
        if texts:
            print(f"📂 Loaded {len(texts)} text samples from {data_dir}")
        else:
            print(f"⚠️  No .txt files in {data_dir}, using placeholder data")
            texts = PLACEHOLDER_TEXTS
    else:
        print("📝 Using placeholder data (no data/ directory, FineWeb unavailable)")
        texts = PLACEHOLDER_TEXTS

    full_dataset = TextChunkDataset(texts, tokenizer, training_config.max_seq_length)

    total = len(full_dataset)
    # Limit validation size to val_samples (default: 64) for speed, matching FineWeb streaming mode
    val_size = min(total - 1, max(1, training_config.val_samples))
    train_size = total - val_size

    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"   Train samples: {len(train_ds)}")
    print(f"   Val samples:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
    )

    return train_loader, val_loader


def build_dataloaders(tokenizer, training_config, model_config):
    """
    Main entry point: try FineWeb first, fall back to local data.

    Returns:
        train_loader, val_loader
    """
    if training_config.use_fineweb:
        try:
            return build_fineweb_dataloaders(tokenizer, training_config, model_config)
        except Exception as e:
            print(f"\n⚠️  FineWeb loading failed: {e}")
            print("   Falling back to local/placeholder data...\n")
            return build_fallback_dataloaders(tokenizer, training_config, model_config)
    else:
        return build_fallback_dataloaders(tokenizer, training_config, model_config)


# ===========================================================================
# Helpers
# ===========================================================================

def _load_texts_from_directory(data_dir):
    """Load all .txt files from a directory, split into chunks."""
    texts = []
    txt_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True))

    for filepath in txt_files:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        if not content:
            continue

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        merged, current = [], ""
        for para in paragraphs:
            if len(current) + len(para) < 2000:
                current = (current + " " + para) if current else para
            else:
                if current:
                    merged.append(current)
                current = para
        if current:
            merged.append(current)
        texts.extend(merged)

    return texts
