"""Flower client for federated DPO alignment and unlearning."""

import logging
from collections import OrderedDict
from typing import Any

import flwr as fl
import numpy as np
import torch
from peft import PeftModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.data.tofu_loader import create_dataloader
from src.unlearn.objectives import get_unlearn_loss, NEEDS_REFERENCE
from src.utils.gpu import log_gpu_memory

logger = logging.getLogger(__name__)


def get_lora_state_dict(model: PeftModel) -> OrderedDict:
    """Extract only LoRA parameters as numpy arrays."""
    state = OrderedDict()
    for name, param in model.named_parameters():
        if "lora_" in name:
            state[name] = param.detach().cpu().numpy()
    return state


def set_lora_state_dict(model: PeftModel, state: OrderedDict) -> None:
    """Set LoRA parameters from numpy arrays."""
    model_state = dict(model.named_parameters())
    for name, value in state.items():
        if name in model_state:
            model_state[name].data.copy_(
                torch.from_numpy(value).to(model_state[name].device)
            )


class FedDPOClient(fl.client.NumPyClient):
    """Flower client for DPO alignment and preference unlearning."""

    def __init__(
        self,
        client_id: int,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        train_data: list[dict[str, Any]],
        forget_data: list[dict[str, Any]],
        retain_data: list[dict[str, Any]],
        config: dict,
        ref_model: PreTrainedModel | None = None,
        is_forget_client: bool = False,
    ):
        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.config = config
        self.ref_model = ref_model
        self.is_forget_client = is_forget_client

        self.fl_cfg = config["fl"]
        self.unlearn_cfg = config["unlearning"]

    def get_parameters(self, config: dict | None = None) -> list[np.ndarray]:
        """Return LoRA parameters as list of numpy arrays."""
        state = get_lora_state_dict(self.model)
        return [v for v in state.values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Set LoRA parameters from list of numpy arrays."""
        state = get_lora_state_dict(self.model)
        keys = list(state.keys())
        new_state = OrderedDict()
        for key, param in zip(keys, parameters):
            new_state[key] = param
        set_lora_state_dict(self.model, new_state)

    def _train_dpo_alignment(self, num_epochs: int) -> dict[str, float]:
        """Phase 1: DPO alignment training on all client data."""
        self.model.train()

        dataloader = create_dataloader(
            self.train_data,
            self.tokenizer,
            batch_size=self.fl_cfg["batch_size"],
            max_length=self.config["model"]["max_length"],
        )

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.unlearn_cfg["learning_rate"],
            weight_decay=self.unlearn_cfg["weight_decay"],
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader))

        total_loss = 0.0
        num_batches = 0
        grad_accum = self.fl_cfg.get("gradient_accumulation_steps", 1)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Standard DPO alignment loss (not reversed)
                loss = self._compute_alignment_loss(batch)
                loss = loss / grad_accum
                loss.backward()

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * grad_accum
                num_batches += 1

            logger.info(
                f"Client {self.client_id} alignment epoch {epoch}: "
                f"loss={epoch_loss / max(1, len(dataloader)):.4f}"
            )
            total_loss += epoch_loss

        return {
            "train_loss": total_loss / max(1, num_batches),
            "num_examples": len(self.train_data),
        }

    def _compute_alignment_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Standard DPO alignment loss (non-reversed, for training)."""
        from src.unlearn.objectives import _get_per_token_log_probs, _sequence_log_prob

        prompt_length = batch["prompt_length"]
        beta = self.unlearn_cfg["beta"]

        chosen_lp, chosen_mask = _get_per_token_log_probs(
            self.model, batch["chosen_input_ids"],
            batch["chosen_attention_mask"], prompt_length
        )
        rejected_lp, rejected_mask = _get_per_token_log_probs(
            self.model, batch["rejected_input_ids"],
            batch["rejected_attention_mask"], prompt_length
        )

        if self.ref_model is not None:
            with torch.no_grad():
                ref_chosen_lp, _ = _get_per_token_log_probs(
                    self.ref_model, batch["chosen_input_ids"].to(self.ref_model.device),
                    batch["chosen_attention_mask"].to(self.ref_model.device), prompt_length
                )
                ref_rejected_lp, _ = _get_per_token_log_probs(
                    self.ref_model, batch["rejected_input_ids"].to(self.ref_model.device),
                    batch["rejected_attention_mask"].to(self.ref_model.device), prompt_length
                )
            ref_chosen_lp = ref_chosen_lp.to(chosen_lp.device)
            ref_rejected_lp = ref_rejected_lp.to(rejected_lp.device)

            pi_c = _sequence_log_prob(chosen_lp, chosen_mask)
            pi_r = _sequence_log_prob(rejected_lp, rejected_mask)
            ref_c = _sequence_log_prob(ref_chosen_lp, chosen_mask)
            ref_r = _sequence_log_prob(ref_rejected_lp, rejected_mask)

            logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
        else:
            from src.unlearn.objectives import _avg_sequence_log_prob
            avg_c = _avg_sequence_log_prob(chosen_lp, chosen_mask)
            avg_r = _avg_sequence_log_prob(rejected_lp, rejected_mask)
            logits = beta * (avg_c - avg_r)

        return -torch.nn.functional.logsigmoid(logits).mean()

    def _train_unlearning(self, num_epochs: int) -> dict[str, float]:
        """Phase 2: Unlearning on forget data (for forget client)."""
        if not self.is_forget_client or not self.forget_data:
            return {"unlearn_loss": 0.0, "num_examples": 0}

        self.model.train()
        objective = self.unlearn_cfg["objective"]

        forget_loader = create_dataloader(
            self.forget_data,
            self.tokenizer,
            batch_size=self.fl_cfg["batch_size"],
            max_length=self.config["model"]["max_length"],
        )

        # Retain data loader for regularization
        retain_loader = None
        if self.retain_data:
            retain_loader = create_dataloader(
                self.retain_data,
                self.tokenizer,
                batch_size=self.fl_cfg["batch_size"],
                max_length=self.config["model"]["max_length"],
            )

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.unlearn_cfg["learning_rate"],
            weight_decay=self.unlearn_cfg["weight_decay"],
        )

        total_loss = 0.0
        num_batches = 0
        grad_accum = self.fl_cfg.get("gradient_accumulation_steps", 1)
        retain_weight = self.unlearn_cfg.get("retain_weight", 1.0)

        retain_iter = iter(retain_loader) if retain_loader else None

        for epoch in range(num_epochs):
            for step, forget_batch in enumerate(forget_loader):
                forget_batch = {
                    k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in forget_batch.items()
                }

                # Unlearning loss on forget data
                loss = get_unlearn_loss(
                    objective=objective,
                    model=self.model,
                    batch=forget_batch,
                    ref_model=self.ref_model,
                    beta=self.unlearn_cfg["beta"],
                    gamma=self.unlearn_cfg.get("gamma", 1.0),
                )

                # Retain regularization
                if retain_iter is not None and retain_weight > 0:
                    try:
                        retain_batch = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(retain_loader)
                        retain_batch = next(retain_iter)

                    retain_batch = {
                        k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                        for k, v in retain_batch.items()
                    }
                    retain_loss = self._compute_alignment_loss(retain_batch)
                    loss = loss + retain_weight * retain_loss

                loss = loss / grad_accum
                loss.backward()

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * grad_accum
                num_batches += 1

            logger.info(
                f"Client {self.client_id} unlearning epoch {epoch} "
                f"({objective}): loss={total_loss / max(1, num_batches):.4f}"
            )

        return {
            "unlearn_loss": total_loss / max(1, num_batches),
            "num_examples": len(self.forget_data),
            "objective": objective,
        }

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[list[np.ndarray], int, dict]:
        """Flower fit: run alignment or unlearning depending on phase."""
        self.set_parameters(parameters)

        phase = config.get("phase", "alignment")
        gpu_id = self.config.get("hardware", {}).get("train_gpu", 0)
        log_gpu_memory(gpu_id, f"Client {self.client_id} {phase} start")

        if phase == "alignment":
            epochs = self.fl_cfg["local_epochs_alignment"]
            metrics = self._train_dpo_alignment(epochs)
        elif phase == "unlearning":
            epochs = self.fl_cfg["local_epochs_unlearning"]
            metrics = self._train_unlearning(epochs)
        else:
            raise ValueError(f"Unknown phase: {phase}")

        log_gpu_memory(gpu_id, f"Client {self.client_id} {phase} end")

        metrics["client_id"] = self.client_id
        return self.get_parameters(), len(self.train_data), metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[float, int, dict]:
        """Flower evaluate (not used in this setup, eval is centralized)."""
        self.set_parameters(parameters)
        return 0.0, len(self.train_data), {}
