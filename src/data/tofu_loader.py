"""TOFU dataset loading and federated partitioning."""

import logging
from typing import Any

import numpy as np
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def load_tofu_dataset(split: str = "full") -> Dataset:
    """Load TOFU dataset from HuggingFace.

    TOFU has splits: full, forget01, forget05, forget10, retain90, retain95, retain99
    """
    ds = load_dataset("locuslab/TOFU", split, split="train")
    logger.info(f"Loaded TOFU '{split}' split: {len(ds)} examples")
    return ds


def partition_iid(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> list[Dataset]:
    """Partition dataset IID across clients."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    splits = np.array_split(indices, num_clients)
    partitions = [dataset.select(s.tolist()) for s in splits]

    for i, p in enumerate(partitions):
        logger.info(f"Client {i}: {len(p)} examples (IID)")
    return partitions


def partition_non_iid(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> list[Dataset]:
    """Partition dataset non-IID using Dirichlet distribution.

    For TOFU, we group by author (first word of question as proxy)
    and distribute groups unevenly across clients.
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)

    # Simple non-IID: Dirichlet allocation of indices
    proportions = rng.dirichlet(np.full(num_clients, alpha), size=1)[0]
    proportions = (proportions * n).astype(int)
    # Fix rounding
    proportions[-1] = n - proportions[:-1].sum()

    indices = rng.permutation(n)
    splits = []
    start = 0
    for count in proportions:
        splits.append(indices[start : start + count].tolist())
        start += count

    partitions = [dataset.select(s) for s in splits]
    for i, p in enumerate(partitions):
        logger.info(f"Client {i}: {len(p)} examples (non-IID, alpha={alpha})")
    return partitions


def create_forget_retain_split(
    partition: Dataset,
    forget_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Split a client's partition into forget and retain sets."""
    n = len(partition)
    n_forget = max(1, int(n * forget_ratio))

    rng = np.random.default_rng(seed)
    forget_indices = rng.choice(n, size=n_forget, replace=False).tolist()
    retain_indices = [i for i in range(n) if i not in set(forget_indices)]

    forget_set = partition.select(forget_indices)
    retain_set = partition.select(retain_indices)

    logger.info(f"Split: {len(forget_set)} forget, {len(retain_set)} retain")
    return forget_set, retain_set


def format_as_preference_pairs(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> list[dict[str, Any]]:
    """Format TOFU data as preference pairs for DPO.

    TOFU has 'question' and 'answer' fields.
    For DPO alignment:
      - chosen = correct answer
      - rejected = "I don't know" / refusal response
    For unlearning (reversed):
      - chosen = "I don't know" (what we want the model to say)
      - rejected = correct answer (what we want it to forget)
    """
    refusal_responses = [
        "I don't have information about that.",
        "I'm not able to answer that question.",
        "I don't know the answer to that.",
        "That information isn't available to me.",
        "I cannot provide an answer to that question.",
    ]

    pairs = []
    for i, example in enumerate(dataset):
        question = example["question"]
        answer = example["answer"]
        refusal = refusal_responses[i % len(refusal_responses)]

        prompt = f"Question: {question}\nAnswer:"

        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length // 2,
            return_tensors="pt",
        )

        chosen_tokens = tokenizer(
            answer,
            truncation=True,
            max_length=max_length // 2,
            return_tensors="pt",
        )

        rejected_tokens = tokenizer(
            refusal,
            truncation=True,
            max_length=max_length // 2,
            return_tensors="pt",
        )

        pairs.append({
            "prompt": prompt,
            "chosen": answer,
            "rejected": refusal,
            "prompt_input_ids": prompt_tokens["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_tokens["attention_mask"].squeeze(0),
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
        })

    return pairs


def collate_preference_pairs(
    batch: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    """Collate function for preference pair batches."""
    prompts = [item["prompt"] for item in batch]
    chosens = [item["chosen"] for item in batch]
    rejecteds = [item["rejected"] for item in batch]

    # Tokenize chosen (prompt + chosen answer)
    chosen_texts = [p + c for p, c in zip(prompts, chosens)]
    chosen_enc = tokenizer(
        chosen_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Tokenize rejected (prompt + rejected answer)
    rejected_texts = [p + r for p, r in zip(prompts, rejecteds)]
    rejected_enc = tokenizer(
        rejected_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Tokenize prompt only (for masking)
    prompt_enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length // 2,
        return_tensors="pt",
    )

    return {
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention_mask": chosen_enc["attention_mask"],
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_enc["attention_mask"],
        "prompt_length": prompt_enc["input_ids"].shape[1],
    }


def create_dataloader(
    pairs: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    max_length: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader from preference pairs."""
    return DataLoader(
        pairs,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_preference_pairs(b, tokenizer, max_length),
    )


def prepare_federated_data(
    config: dict,
    tokenizer: PreTrainedTokenizer,
) -> dict[str, Any]:
    """Prepare all federated data splits.

    Returns dict with:
      - client_data: list of (train_pairs, forget_pairs, retain_pairs) per client
      - global_eval: evaluation dataset
    """
    data_cfg = config["data"]
    fl_cfg = config["fl"]

    # Load full dataset
    full_ds = load_tofu_dataset("full")

    # Partition across clients
    if data_cfg.get("non_iid", False):
        partitions = partition_non_iid(
            full_ds,
            fl_cfg["num_clients"],
            alpha=data_cfg.get("non_iid_alpha", 0.5),
            seed=data_cfg.get("seed", 42),
        )
    else:
        partitions = partition_iid(
            full_ds,
            fl_cfg["num_clients"],
            seed=data_cfg.get("seed", 42),
        )

    client_data = []
    for cid, partition in enumerate(partitions):
        # Create forget/retain split for the forget client
        if cid == fl_cfg["forget_client_id"]:
            forget_set, retain_set = create_forget_retain_split(
                partition,
                forget_ratio=data_cfg.get("forget_ratio", 0.1),
                seed=data_cfg.get("seed", 42),
            )
            forget_pairs = format_as_preference_pairs(
                forget_set, tokenizer, config["model"]["max_length"]
            )
            retain_pairs = format_as_preference_pairs(
                retain_set, tokenizer, config["model"]["max_length"]
            )
        else:
            forget_pairs = []
            retain_pairs = format_as_preference_pairs(
                partition, tokenizer, config["model"]["max_length"]
            )

        train_pairs = format_as_preference_pairs(
            partition, tokenizer, config["model"]["max_length"]
        )

        client_data.append({
            "train": train_pairs,
            "forget": forget_pairs,
            "retain": retain_pairs,
            "client_id": cid,
            "is_forget_client": cid == fl_cfg["forget_client_id"],
        })

    # Load eval splits
    forget_eval = load_tofu_dataset("forget10")
    retain_eval = load_tofu_dataset("retain90")

    return {
        "client_data": client_data,
        "forget_eval": format_as_preference_pairs(
            forget_eval, tokenizer, config["model"]["max_length"]
        ),
        "retain_eval": format_as_preference_pairs(
            retain_eval, tokenizer, config["model"]["max_length"]
        ),
    }
