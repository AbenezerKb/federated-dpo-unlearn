"""Model loading with LoRA adapter configuration."""

import logging
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.gpu import get_device

logger = logging.getLogger(__name__)

TASK_TYPE_MAP = {
    "CAUSAL_LM": TaskType.CAUSAL_LM,
}


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load tokenizer with proper padding configuration."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_base_model(
    model_name: str,
    device: torch.device,
    dtype: str = "bfloat16",
) -> AutoModelForCausalLM:
    """Load base model without LoRA."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map={"": device},
        trust_remote_code=False,
    )
    model.config.use_cache = False
    return model


def create_lora_config(lora_cfg: dict) -> LoraConfig:
    """Create LoRA configuration from config dict."""
    return LoraConfig(
        r=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get(
            "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
        ),
        task_type=TASK_TYPE_MAP.get(
            lora_cfg.get("task_type", "CAUSAL_LM"), TaskType.CAUSAL_LM
        ),
        bias="none",
    )


def load_model_with_lora(
    config: dict,
    device: torch.device | None = None,
) -> PeftModel:
    """Load model with fresh LoRA adapters."""
    model_cfg = config["model"]
    lora_cfg = config["lora"]

    if device is None:
        device = get_device(config.get("hardware", {}).get("train_gpu", 0))

    logger.info(f"Loading {model_cfg['name']} on {device}")
    base_model = load_base_model(model_cfg["name"], device, model_cfg.get("dtype", "bfloat16"))

    lora_config = create_lora_config(lora_cfg)
    peft_model = get_peft_model(base_model, lora_config)

    trainable, total = peft_model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA params: {trainable:,} trainable / {total:,} total "
        f"({100 * trainable / total:.2f}%)"
    )
    return peft_model


def load_reference_model(
    config: dict,
    device: torch.device | None = None,
) -> AutoModelForCausalLM:
    """Load frozen reference model for DPO/NPO objectives."""
    model_cfg = config["model"]
    if device is None:
        device = get_device(config.get("hardware", {}).get("ref_model_gpu", 1))

    logger.info(f"Loading reference model on {device}")
    model = load_base_model(model_cfg["name"], device, model_cfg.get("dtype", "bfloat16"))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_checkpoint(
    checkpoint_path: str | Path,
    config: dict | None = None,
    device: torch.device | None = None,
) -> PeftModel:
    """Load model + LoRA adapter weights from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if config is None:
        adapter_config_path = checkpoint_path / "adapter_config.json"
        if not adapter_config_path.exists():
            raise FileNotFoundError(f"No adapter_config.json in {checkpoint_path}")
        import json
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg["base_model_name_or_path"]
    else:
        base_model_name = config["model"]["name"]

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_model = load_base_model(base_model_name, device)
    model = PeftModel.from_pretrained(
        base_model, str(checkpoint_path), is_trainable=True
    )
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return model


def save_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    save_path: str | Path,
) -> None:
    """Save LoRA adapter weights and tokenizer."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if isinstance(model, PeftModel):
        model.save_pretrained(str(save_path))
    else:
        model.save_pretrained(str(save_path))

    tokenizer.save_pretrained(str(save_path))
    logger.info(f"Saved checkpoint to {save_path}")
