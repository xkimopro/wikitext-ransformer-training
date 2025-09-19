from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer
from datasets import load_dataset


def _collate_fixed_seq(batch):
    # Items are fixed-length sequences already; no padding mask needed.
    input_ids = [item["input_ids"] for item in batch]
    input_ids_tensor = torch.as_tensor(input_ids, dtype=torch.long)
    # We return ONLY input_ids to avoid any downstream masking overhead.
    return {"input_ids": input_ids_tensor}


def get_dataloaders(config):
    """
    Build tokenized, fixed-length sequence DataLoaders purely from YAML config.
    No attention mask is produced (sequences are fixed length; we use a causal mask in the model).
    """
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(
        config.data.dataset_name,
        config.data.dataset_config,
        streaming=bool(config.data.streaming),
    )

    # If missing validation split, optionally create it
    if "validation" not in raw and bool(config.data.create_val_split_if_missing):
        split = raw["train"].train_test_split(
            test_size=float(config.data.val_split_fraction),
            seed=int(config.seed)
        )
        raw = {"train": split["train"], "validation": split["test"]}

    column_names = raw["train"].column_names
    text_column = config.data.text_column
    seq_len = int(config.data.seq_len)

    def process_texts(examples):
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=seq_len,
            padding=False,
            return_attention_mask=False,
        )

        tokens = []
        for seq in tokenized["input_ids"]:
            tokens.extend(seq)

        total_length = len(tokens)
        if total_length < seq_len:
            return {"input_ids": []}

        total_length = (total_length // seq_len) * seq_len
        input_ids = [tokens[i : i + seq_len] for i in range(0, total_length, seq_len)]
        return {"input_ids": input_ids}

    lm = raw.map(
        process_texts,
        batched=True,
        remove_columns=column_names,
    )

    train_ds = lm["train"]
    eval_ds = lm["validation"]

    if bool(config.data.streaming):
        train_ds = train_ds.shuffle(
            buffer_size=int(config.data.shuffle_buffer), seed=int(config.seed)
        )

    # Dataloader options from YAML
    num_workers = int(config.loader.num_workers)
    persistent_workers = bool(config.loader.persistent_workers) and num_workers > 0
    prefetch_factor = int(config.loader.prefetch_factor) if num_workers > 0 else None
    pin_memory = bool(config.loader.pin_memory)
    drop_last = bool(config.loader.drop_last)
    batch_size = int(config.train.batch_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=_collate_fixed_seq,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
    )

    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        collate_fn=_collate_fixed_seq,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=drop_last,
    )

    return train_loader, eval_loader, tokenizer
