"""Run a single phase of the federated DPO unlearning experiment.

Uses a manual FL loop instead of Flower's Ray-based simulation to
support multi-GPU device placement (Ray isolates CUDA visibility
per worker, breaking cross-GPU setups).
"""

import argparse
import json
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def fedavg_aggregate(
    client_params: list[list[np.ndarray]],
    client_sizes: list[int],
) -> list[np.ndarray]:
    """Weighted average of client parameters (FedAvg)."""
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]

    avg_params = []
    for param_idx in range(len(client_params[0])):
        weighted = sum(
            w * client_params[cid][param_idx]
            for cid, w in enumerate(weights)
        )
        avg_params.append(weighted)
    return avg_params


def run_alignment(config: dict, output_dir: str):
    """Phase 1: Federated DPO alignment training."""
    from src.data.tofu_loader import prepare_federated_data
    from src.fl.flower_client import (
        FedDPOClient,
        get_lora_state_dict,
        set_lora_state_dict,
    )
    from src.models.model_loader import (
        load_model_with_lora,
        load_reference_model,
        load_tokenizer,
        save_checkpoint,
    )
    from src.utils.gpu import assign_devices, log_gpu_memory

    devices = assign_devices(config)
    tokenizer = load_tokenizer(config["model"]["name"])
    fed_data = prepare_federated_data(config, tokenizer)

    model = load_model_with_lora(config, device=devices["train"])
    ref_model = load_reference_model(config, device=devices["ref_model"])

    fl_cfg = config["fl"]
    num_rounds = fl_cfg["num_rounds_alignment"]
    num_clients = fl_cfg["num_clients"]

    # Create clients (all share same model object, params swapped per client)
    clients = []
    for cid in range(num_clients):
        cd = fed_data["client_data"][cid]
        clients.append(FedDPOClient(
            client_id=cid,
            model=model,
            tokenizer=tokenizer,
            train_data=cd["train"],
            forget_data=cd["forget"],
            retain_data=cd["retain"],
            config=config,
            ref_model=ref_model,
            is_forget_client=cd["is_forget_client"],
        ))

    # Get initial global params
    global_params = clients[0].get_parameters()

    for rnd in range(1, num_rounds + 1):
        logger.info(f"=== Alignment Round {rnd}/{num_rounds} ===")
        log_gpu_memory(devices["train"].index or 0, f"Round {rnd} start")

        round_params = []
        round_sizes = []

        for client in clients:
            client.set_parameters(global_params)
            params, num_examples, metrics = client.fit(
                global_params,
                config={"phase": "alignment", "server_round": rnd},
            )
            round_params.append(params)
            round_sizes.append(num_examples)
            logger.info(
                f"  Client {client.client_id}: "
                f"loss={metrics.get('train_loss', 0):.4f}, "
                f"examples={num_examples}"
            )

        global_params = fedavg_aggregate(round_params, round_sizes)
        logger.info(f"Round {rnd} aggregation complete")

    # Set final aggregated params
    clients[0].set_parameters(global_params)

    os.makedirs(output_dir, exist_ok=True)
    save_checkpoint(model, tokenizer, output_dir)
    logger.info(f"Alignment checkpoint saved: {output_dir}")


def run_unlearning(
    config: dict,
    objective: str,
    alignment_checkpoint: str,
    output_dir: str,
):
    """Phase 2: Federated unlearning with specified objective."""
    from src.data.tofu_loader import prepare_federated_data
    from src.fl.flower_client import FedDPOClient
    from src.fl.gradient_shielding import shield_gradients
    from src.models.model_loader import (
        load_checkpoint,
        load_reference_model,
        load_tokenizer,
        save_checkpoint,
    )
    from src.utils.gpu import assign_devices, log_gpu_memory

    devices = assign_devices(config)
    tokenizer = load_tokenizer(config["model"]["name"])
    fed_data = prepare_federated_data(config, tokenizer)

    model = load_checkpoint(
        alignment_checkpoint, config=config, device=devices["train"]
    )
    ref_model = load_reference_model(config, device=devices["ref_model"])

    config["unlearning"]["objective"] = objective

    fl_cfg = config["fl"]
    gs_cfg = config.get("gradient_shielding", {})
    num_rounds = fl_cfg["num_rounds_unlearning"]
    num_clients = fl_cfg["num_clients"]
    forget_client_id = fl_cfg["forget_client_id"]

    clients = []
    for cid in range(num_clients):
        cd = fed_data["client_data"][cid]
        clients.append(FedDPOClient(
            client_id=cid,
            model=model,
            tokenizer=tokenizer,
            train_data=cd["train"],
            forget_data=cd["forget"],
            retain_data=cd["retain"],
            config=config,
            ref_model=ref_model,
            is_forget_client=cd["is_forget_client"],
        ))

    global_params = clients[0].get_parameters()
    shielding_metrics = []

    for rnd in range(1, num_rounds + 1):
        logger.info(
            f"=== Unlearning Round {rnd}/{num_rounds} ({objective}) ==="
        )
        log_gpu_memory(devices["train"].index or 0, f"Round {rnd} start")

        round_params = []
        round_sizes = []
        forget_idx = None

        for i, client in enumerate(clients):
            client.set_parameters(global_params)
            params, num_examples, metrics = client.fit(
                global_params,
                config={"phase": "unlearning", "server_round": rnd},
            )
            round_params.append(params)
            round_sizes.append(num_examples)

            if client.client_id == forget_client_id:
                forget_idx = i

            logger.info(
                f"  Client {client.client_id}: "
                f"loss={metrics.get('unlearn_loss', metrics.get('train_loss', 0)):.4f}, "
                f"examples={num_examples}"
            )

        # Apply gradient shielding if enabled
        if gs_cfg.get("enabled", False) and forget_idx is not None:
            logger.info("Applying gradient shielding...")

            # Compute deltas (client params - global params)
            def flatten_params(params):
                return np.concatenate([p.flatten() for p in params])

            def unflatten_params(flat, shapes):
                arrays = []
                offset = 0
                for shape in shapes:
                    size = int(np.prod(shape))
                    arrays.append(
                        flat[offset:offset + size].reshape(shape)
                    )
                    offset += size
                return arrays

            shapes = [p.shape for p in global_params]
            global_flat = flatten_params(global_params)

            forget_delta = (
                flatten_params(round_params[forget_idx]) - global_flat
            )
            retained_deltas = [
                flatten_params(round_params[i]) - global_flat
                for i in range(len(clients))
                if i != forget_idx
            ]

            shielded_delta, metrics = shield_gradients(
                forget_delta,
                retained_deltas,
                subspace_dim=gs_cfg.get("subspace_dim", 64),
                projection_strength=gs_cfg.get("projection_strength", 1.0),
            )
            shielding_metrics.append({"round": rnd, **metrics})

            # Replace forget client params with shielded version
            round_params[forget_idx] = unflatten_params(
                global_flat + shielded_delta, shapes
            )

        global_params = fedavg_aggregate(round_params, round_sizes)
        logger.info(f"Round {rnd} aggregation complete")

    # Set final params and save
    clients[0].set_parameters(global_params)

    unlearned_dir = os.path.join(output_dir, "unlearned_model")
    os.makedirs(unlearned_dir, exist_ok=True)
    save_checkpoint(model, tokenizer, unlearned_dir)

    if shielding_metrics:
        metrics_path = os.path.join(output_dir, "shielding_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(shielding_metrics, f, indent=2)

    logger.info(
        f"Unlearning ({objective}) complete. Model saved: {unlearned_dir}"
    )


def run_eval(
    config: dict,
    objective: str,
    model_checkpoint: str,
    output_dir: str,
):
    """Phase 3: Evaluate unlearned model on TOFU metrics."""
    from src.data.tofu_loader import prepare_federated_data
    from src.eval.tofu_metrics import run_full_evaluation
    from src.models.model_loader import load_checkpoint, load_tokenizer
    from src.utils.gpu import assign_devices

    devices = assign_devices(config)
    tokenizer = load_tokenizer(config["model"]["name"])
    fed_data = prepare_federated_data(config, tokenizer)

    model = load_checkpoint(
        model_checkpoint, config=config, device=devices["eval"]
    )

    results = run_full_evaluation(
        model=model,
        tokenizer=tokenizer,
        forget_data=fed_data["forget_eval"],
        retain_data=fed_data["retain_eval"],
        device=devices["eval"],
        objective=objective,
    )

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results saved: {results_path}")
    logger.info(json.dumps(results, indent=2))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase",
        required=True,
        choices=["alignment", "unlearning", "eval"],
    )
    parser.add_argument("--objective", default="dpo")
    parser.add_argument("--alignment-checkpoint", default=None)
    parser.add_argument("--model-checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.phase == "alignment":
        run_alignment(config, args.output_dir)
    elif args.phase == "unlearning":
        if not args.alignment_checkpoint:
            raise ValueError("--alignment-checkpoint required for unlearning")
        run_unlearning(
            config, args.objective,
            args.alignment_checkpoint, args.output_dir,
        )
    elif args.phase == "eval":
        if not args.model_checkpoint:
            raise ValueError("--model-checkpoint required for eval")
        run_eval(
            config, args.objective,
            args.model_checkpoint, args.output_dir,
        )


if __name__ == "__main__":
    main()
