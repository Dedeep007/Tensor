"""
OpenOrca Instruction Dataset for OPTIMUS MoE
===============================================
Loads Open-Orca/OpenOrca from HuggingFace, formats samples into
chat template, tokenizes, and applies loss masking so the model
only learns to generate assistant responses.

Chat Template (Qwen/ChatML format):
    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    {response}<|im_end|>

The Qwen2.5 tokenizer natively supports <|im_start|> and <|im_end|> tokens.
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Add parent directories for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dataset import get_tokenizer


def format_chat_template(system_prompt, question, response):
    """
    Format a single OpenOrca sample into ChatML format.

    Returns:
        full_text: The complete formatted conversation
        prompt_text: Everything before the assistant's response (for loss masking)
    """
    # Build the prompt (system + user turns)
    prompt_text = ""
    if system_prompt and system_prompt.strip():
        prompt_text += f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"
    prompt_text += f"<|im_start|>user\n{question.strip()}<|im_end|>\n"
    prompt_text += "<|im_start|>assistant\n"

    # Full text includes the response
    full_text = prompt_text + f"{response.strip()}<|im_end|>"

    return full_text, prompt_text


class OpenOrcaDataset(Dataset):
    """
    Map-style dataset for Open-Orca/OpenOrca instruction tuning.

    Loads samples into memory, tokenizes with chat template formatting,
    and applies loss masking so only assistant response tokens contribute
    to the training loss.

    Args:
        tokenizer: Qwen2.5 tokenizer
        max_length: Maximum sequence length (default 1024)
        max_samples: Number of samples to load (None = all)
        mask_prompt: If True, set labels to -100 for non-response tokens
        split: Which split to load ("train")
    """

    def __init__(
        self,
        tokenizer,
        max_length=1024,
        max_samples=100_000,
        mask_prompt=True,
        split="train",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt = mask_prompt
        self.samples = []

        print(f"   Loading OpenOrca dataset (max_samples={max_samples})...")
        self._load_dataset(max_samples, split)
        print(f"   Loaded {len(self.samples)} samples")

    def _load_dataset(self, max_samples, split):
        """Load and preprocess OpenOrca samples."""
        from datasets import load_dataset

        ds = load_dataset(
            "Open-Orca/OpenOrca",
            split=split,
            streaming=True,
        )

        count = 0
        skipped = 0
        for sample in ds:
            if max_samples is not None and count >= max_samples:
                break

            system_prompt = sample.get("system_prompt", "")
            question = sample.get("question", "")
            response = sample.get("response", "")

            # Skip empty or very short samples
            if not question.strip() or not response.strip():
                skipped += 1
                continue
            if len(response.strip()) < 5:
                skipped += 1
                continue

            # Build the chat template once here so we can filter out examples
            # whose prompt alone already fills the sequence window.
            full_text, prompt_text = format_chat_template(system_prompt, question, response)
            prompt_encoding = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            prompt_len = len(prompt_encoding["input_ids"])
            if prompt_len >= self.max_length - 1:
                skipped += 1
                continue

            self.samples.append({
                "system_prompt": system_prompt,
                "question": question,
                "response": response,
            })
            count += 1

            # Progress logging
            if count % 10_000 == 0:
                print(f"     Loaded {count:,} samples...")

        if skipped > 0:
            print(f"   Skipped {skipped} empty/short samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        full_text, prompt_text = format_chat_template(
            sample["system_prompt"],
            sample["question"],
            sample["response"],
        )

        # Tokenize full text
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = full_encoding["input_ids"].squeeze(0)
        attention_mask = full_encoding["attention_mask"].squeeze(0)

        # Build labels
        labels = input_ids.clone()

        # Mask padding tokens
        labels[attention_mask == 0] = -100

        # Mask prompt tokens (only train on assistant response)
        if self.mask_prompt:
            # Tokenize just the prompt to find where the response starts
            prompt_encoding = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            prompt_len = len(prompt_encoding["input_ids"])

            # Mask all tokens up to (and including) the prompt
            if prompt_len > 0:
                labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_instruct_dataloaders(tokenizer, config):
    """
    Build train and validation DataLoaders for instruction fine-tuning.

    Args:
        tokenizer: Qwen2.5 tokenizer
        config: InstructFinetuneConfig

    Returns:
        train_loader, val_loader
    """
    print("\n📚 Building instruction fine-tuning dataset...")

    dataset = OpenOrcaDataset(
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
        max_samples=config.max_samples,
        mask_prompt=config.mask_prompt,
    )

    # Train/val split
    total = len(dataset)
    val_size = min(int(total * config.val_split), config.val_samples)
    val_size = max(val_size, 1)  # At least 1 val sample
    train_size = total - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"   Train samples: {len(train_ds):,}")
    print(f"   Val samples:   {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,          # Windows compatibility
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader
