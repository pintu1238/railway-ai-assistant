"""
Railway AI Assistant — Fine-Tuning Script (Local GPU)
Model: T5-small | Dataset: data/dataset.csv
Run: python fine_tuning.py
"""

import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "model_name": "t5-small",
    "dataset_path": "data/dataset.csv",
    "output_dir": "railway_model",
    "max_input_length": 256,
    "max_target_length": 128,
    "batch_size": 8,
    "epochs": 5,
    "learning_rate": 3e-4,
    "warmup_steps": 100,
    "gradient_accumulation_steps": 4,
    "val_split": 0.1,
    "seed": 42,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class RailwayDataset(Dataset):
    def __init__(self, data, tokenizer, max_input_len, max_target_len):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Detect column names (handle both formats)
        if "input_text" in row:
            input_col, target_col = "input_text", "target_response"
        else:
            input_col, target_col = "conversation_history", "assistant_response"

        input_text = f"railway query: {str(row[input_col])}"
        target_text = str(row[target_col])

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding in loss

        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels,
        }


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler, accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(loader)


def eval_epoch(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────
# INFERENCE TEST
# ─────────────────────────────────────────────
def test_inference(model, tokenizer, queries):
    model.eval()
    print("\n🧪 Inference Test:")
    print("─" * 60)
    for query in queries:
        input_text = f"railway query: {query}"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=CONFIG["max_input_length"],
            truncation=True,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=4,
                early_stopping=True,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Q: {query}")
        print(f"  A: {response}")
        print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    torch.manual_seed(CONFIG["seed"])

    print("\n" + "=" * 60)
    print("  🚂 Railway AI — Fine-Tuning (T5-small)")
    print("=" * 60)

    # Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_csv(CONFIG["dataset_path"])
    print(f"   Total rows: {len(df)}")

    # Drop rows with NaN
    df = df.dropna(subset=[df.columns[1], df.columns[2]])
    print(f"   After cleaning: {len(df)} rows")

    # Train/val split
    train_df, val_df = train_test_split(
        df, test_size=CONFIG["val_split"], random_state=CONFIG["seed"]
    )
    print(f"   Train: {len(train_df)} | Val: {len(val_df)}")

    # Load tokenizer and model
    print(f"\n🤖 Loading {CONFIG['model_name']}...")
    tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"])
    model = T5ForConditionalGeneration.from_pretrained(CONFIG["model_name"])
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Datasets and loaders
    train_dataset = RailwayDataset(
        train_df, tokenizer,
        CONFIG["max_input_length"], CONFIG["max_target_length"]
    )
    val_dataset = RailwayDataset(
        val_df, tokenizer,
        CONFIG["max_input_length"], CONFIG["max_target_length"]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)
    total_steps = (len(train_loader) // CONFIG["gradient_accumulation_steps"]) * CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Training
    print(f"\n🏋️  Training for {CONFIG['epochs']} epochs...")
    print("─" * 60)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            scaler, CONFIG["gradient_accumulation_steps"]
        )
        val_loss = eval_epoch(model, val_loader)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"  Epoch {epoch}/{CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(CONFIG["output_dir"], exist_ok=True)
            model.save_pretrained(CONFIG["output_dir"])
            tokenizer.save_pretrained(CONFIG["output_dir"])
            print(f"    💾 Best model saved (val_loss={val_loss:.4f})")

    # Save training history
    with open(os.path.join(CONFIG["output_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Quick inference test
    test_queries = [
        "User: Train to Manchester tomorrow morning",
        "User: Train to Edinburgh\nAssistant: Any requirements?\nUser: I have a bike",
        "User: Cheapest train to Leeds tonight",
        "User: I need to reach Heathrow urgently",
    ]
    test_inference(model, tokenizer, test_queries)

    print("=" * 60)
    print(f"  ✅ Training complete!")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Model saved   : ./{CONFIG['output_dir']}/")
    print("=" * 60)


if __name__ == "__main__":
    main()