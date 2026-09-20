"""GPU assignment and memory monitoring utilities."""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(gpu_id: int) -> torch.device:
    """Get torch device for a given GPU id."""
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        return torch.device(f"cuda:{gpu_id}")
    logger.warning(f"GPU {gpu_id} not available, falling back to CPU")
    return torch.device("cpu")


def get_gpu_memory(gpu_id: int) -> dict[str, float]:
    """Get GPU memory usage in GB."""
    if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
        return {"total": 0, "used": 0, "free": 0}

    total = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
    reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
    allocated = torch.cuda.memory_allocated(gpu_id) / 1e9

    return {
        "total": round(total, 2),
        "reserved": round(reserved, 2),
        "allocated": round(allocated, 2),
        "free": round(total - reserved, 2),
    }


def log_gpu_memory(gpu_id: int, label: str = "") -> None:
    """Log current GPU memory usage."""
    mem = get_gpu_memory(gpu_id)
    prefix = f"[{label}] " if label else ""
    logger.info(
        f"{prefix}GPU {gpu_id}: "
        f"{mem['allocated']:.1f}GB allocated / "
        f"{mem['total']:.1f}GB total "
        f"({mem['free']:.1f}GB free)"
    )


def clear_gpu_cache(gpu_id: int | None = None) -> None:
    """Clear CUDA cache for a specific GPU or all GPUs."""
    if gpu_id is not None:
        with torch.cuda.device(gpu_id):
            torch.cuda.empty_cache()
    else:
        torch.cuda.empty_cache()


def assign_devices(config: dict) -> dict[str, torch.device]:
    """Assign devices based on config hardware section."""
    hw = config.get("hardware", {})
    return {
        "train": get_device(hw.get("train_gpu", 0)),
        "eval": get_device(hw.get("eval_gpu", 1)),
        "ref_model": get_device(hw.get("ref_model_gpu", 1)),
    }
