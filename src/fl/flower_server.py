"""Flower server with FedAvg and optional FUGAS gradient shielding."""

import logging
from typing import Any

import flwr as fl
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.fl.gradient_shielding import shield_gradients

logger = logging.getLogger(__name__)


class FedAvgWithGradientShielding(FedAvg):
    """FedAvg with optional FUGAS-style gradient shielding.

    During unlearning phase, the forget client's parameter delta is
    projected onto a compatibility subspace derived from retained
    clients' deltas before aggregation.
    """

    def __init__(
        self,
        forget_client_id: int = 0,
        shielding_enabled: bool = False,
        subspace_dim: int = 64,
        projection_strength: float = 1.0,
        phase: str = "alignment",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.forget_client_id = forget_client_id
        self.shielding_enabled = shielding_enabled
        self.subspace_dim = subspace_dim
        self.projection_strength = projection_strength
        self.phase = phase
        self._current_round_params: list[np.ndarray] | None = None
        self.shielding_metrics: list[dict] = []

    def set_phase(self, phase: str) -> None:
        """Switch between alignment and unlearning phases."""
        self.phase = phase
        logger.info(f"Server phase set to: {phase}")

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ):
        """Configure fit with phase information."""
        self._current_round_params = parameters_to_ndarrays(parameters)

        configs = super().configure_fit(server_round, parameters, client_manager)
        updated = []
        for client_proxy, fit_ins in configs:
            fit_ins.config["phase"] = self.phase
            fit_ins.config["server_round"] = server_round
            updated.append((client_proxy, fit_ins))
        return updated

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate with optional gradient shielding during unlearning."""

        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} failures")

        if not results:
            return None, {}

        # Standard aggregation during alignment phase
        if self.phase == "alignment" or not self.shielding_enabled:
            return super().aggregate_fit(server_round, results, failures)

        # Unlearning phase with gradient shielding
        logger.info(
            f"Round {server_round}: Applying gradient shielding "
            f"(forget_client={self.forget_client_id})"
        )

        # Separate forget and retained client results
        forget_results = []
        retained_results = []

        for client_proxy, fit_res in results:
            # Client ID from metrics
            cid = fit_res.metrics.get("client_id", -1)
            if cid == self.forget_client_id:
                forget_results.append((client_proxy, fit_res))
            else:
                retained_results.append((client_proxy, fit_res))

        if not forget_results:
            # No forget client in this round, standard aggregation
            return super().aggregate_fit(server_round, results, failures)

        # Compute parameter deltas
        current_params = self._current_round_params
        if current_params is None:
            return super().aggregate_fit(server_round, results, failures)

        # Flatten deltas for each client
        def get_delta(fit_res: FitRes) -> np.ndarray:
            client_params = parameters_to_ndarrays(fit_res.parameters)
            flat_delta = np.concatenate([
                (cp - gp).flatten()
                for cp, gp in zip(client_params, current_params)
            ])
            return flat_delta

        def unflatten_delta(
            flat_delta: np.ndarray,
            shapes: list[tuple],
        ) -> list[np.ndarray]:
            arrays = []
            offset = 0
            for shape in shapes:
                size = int(np.prod(shape))
                arrays.append(flat_delta[offset:offset + size].reshape(shape))
                offset += size
            return arrays

        # Get shapes from current params
        shapes = [p.shape for p in current_params]

        # Compute deltas
        retained_deltas = [get_delta(fr) for _, fr in retained_results]
        forget_delta = get_delta(forget_results[0][1])

        # Apply gradient shielding
        shielded_delta, metrics = shield_gradients(
            forget_delta,
            retained_deltas,
            subspace_dim=self.subspace_dim,
            projection_strength=self.projection_strength,
        )
        self.shielding_metrics.append({
            "round": server_round,
            **metrics,
        })

        # Reconstruct shielded parameters for forget client
        shielded_params = [
            gp + dp
            for gp, dp in zip(current_params, unflatten_delta(shielded_delta, shapes))
        ]

        # Replace forget client's parameters with shielded version
        shielded_fit_res = FitRes(
            status=forget_results[0][1].status,
            parameters=ndarrays_to_parameters(shielded_params),
            num_examples=forget_results[0][1].num_examples,
            metrics=forget_results[0][1].metrics,
        )

        # Combine shielded forget + retained and aggregate normally
        all_results = retained_results + [(forget_results[0][0], shielded_fit_res)]
        return super().aggregate_fit(server_round, all_results, failures)
