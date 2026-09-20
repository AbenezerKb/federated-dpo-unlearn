"""Unlearning objectives: DPO, NPO, SimNPO, SimPO, FUGAS.

All losses operate on preference pairs (prompt + chosen, prompt + rejected)
and return a scalar loss to be backpropagated.

For unlearning, the semantics are reversed from alignment:
  - Alignment: increase P(chosen), decrease P(rejected)
  - Unlearning: decrease P(chosen/known), increase P(rejected/refusal)
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel


def _get_per_token_log_probs(
    model: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    prompt_length: int,
) -> Tensor:
    """Compute per-token log probabilities for the response portion.

    Args:
        model: Language model
        input_ids: (batch, seq_len) full sequence (prompt + response)
        attention_mask: (batch, seq_len)
        prompt_length: number of prompt tokens to skip

    Returns:
        (batch, response_len) log probabilities for each response token
    """
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits  # (batch, seq_len, vocab)

    # Shift: predict next token
    shift_logits = logits[:, :-1, :]  # (batch, seq_len-1, vocab)
    shift_labels = input_ids[:, 1:]   # (batch, seq_len-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask: only response tokens (after prompt)
    response_mask = attention_mask[:, 1:].clone()
    if prompt_length > 0:
        response_mask[:, :prompt_length - 1] = 0

    token_log_probs = token_log_probs * response_mask
    return token_log_probs, response_mask


def _sequence_log_prob(
    token_log_probs: Tensor,
    mask: Tensor,
) -> Tensor:
    """Sum log probs over sequence (masked)."""
    return (token_log_probs * mask).sum(dim=-1)


def _avg_sequence_log_prob(
    token_log_probs: Tensor,
    mask: Tensor,
) -> Tensor:
    """Average log probs over sequence length (masked)."""
    lengths = mask.sum(dim=-1).clamp(min=1)
    return (token_log_probs * mask).sum(dim=-1) / lengths


def dpo_unlearn_loss(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    batch: dict[str, Tensor],
    beta: float = 0.1,
) -> Tensor:
    """DPO unlearning loss (reversed preferences).

    Standard DPO: maximize log P(chosen) - log P(rejected)
    Unlearning DPO: maximize log P(rejected) - log P(chosen)
    i.e., push model away from the known answer toward refusal.

    L = -log sigmoid(beta * ((log pi(r) - log ref(r)) - (log pi(c) - log ref(c))))
    """
    prompt_length = batch["prompt_length"]

    # Model log probs
    chosen_lp, chosen_mask = _get_per_token_log_probs(
        model, batch["chosen_input_ids"], batch["chosen_attention_mask"], prompt_length
    )
    rejected_lp, rejected_mask = _get_per_token_log_probs(
        model, batch["rejected_input_ids"], batch["rejected_attention_mask"], prompt_length
    )

    # Reference model log probs
    with torch.no_grad():
        ref_chosen_lp, _ = _get_per_token_log_probs(
            ref_model, batch["chosen_input_ids"], batch["chosen_attention_mask"], prompt_length
        )
        ref_rejected_lp, _ = _get_per_token_log_probs(
            ref_model, batch["rejected_input_ids"], batch["rejected_attention_mask"], prompt_length
        )

    # Sequence-level log probs
    pi_chosen = _sequence_log_prob(chosen_lp, chosen_mask)
    pi_rejected = _sequence_log_prob(rejected_lp, rejected_mask)
    ref_chosen = _sequence_log_prob(ref_chosen_lp, chosen_mask)
    ref_rejected = _sequence_log_prob(ref_rejected_lp, rejected_mask)

    # Log ratios
    log_ratio_chosen = pi_chosen - ref_chosen
    log_ratio_rejected = pi_rejected - ref_rejected

    # Reversed DPO: prefer rejected over chosen
    logits = beta * (log_ratio_rejected - log_ratio_chosen)
    loss = -F.logsigmoid(logits).mean()

    return loss


def npo_loss(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    batch: dict[str, Tensor],
    beta: float = 0.1,
) -> Tensor:
    """Negative Preference Optimization loss.

    Bounded gradient ascent on forget data:
    L = -beta * mean(log pi(forget) - log ref(forget))

    This pushes the model's probability on forget data below the reference.
    """
    prompt_length = batch["prompt_length"]

    chosen_lp, chosen_mask = _get_per_token_log_probs(
        model, batch["chosen_input_ids"], batch["chosen_attention_mask"], prompt_length
    )

    with torch.no_grad():
        ref_chosen_lp, _ = _get_per_token_log_probs(
            ref_model, batch["chosen_input_ids"], batch["chosen_attention_mask"], prompt_length
        )

    pi_seq = _sequence_log_prob(chosen_lp, chosen_mask)
    ref_seq = _sequence_log_prob(ref_chosen_lp, chosen_mask)

    # Negative = gradient ascent, bounded by reference
    loss = -beta * (pi_seq - ref_seq).mean()

    return loss


def simnpo_loss(
    model: PreTrainedModel,
    batch: dict[str, Tensor],
    beta: float = 0.1,
) -> Tensor:
    """SimNPO: Reference-free Negative Preference Optimization.

    No reference model needed. Uses average log probability as implicit reward.
    L = -log sigmoid(-beta * avg_log_prob(forget))

    Pushes average log probability of forget tokens down.
    """
    prompt_length = batch["prompt_length"]

    chosen_lp, chosen_mask = _get_per_token_log_probs(
        model, batch["chosen_input_ids"], batch["chosen_attention_mask"], prompt_length
    )

    avg_lp = _avg_sequence_log_prob(chosen_lp, chosen_mask)

    # Push average log prob down
    loss = -F.logsigmoid(-beta * avg_lp).mean()

    return loss


def simpo_loss(
    model: PreTrainedModel,
    batch: dict[str, Tensor],
    beta: float = 0.1,
    gamma: float = 1.0,
) -> Tensor:
    """SimPO unlearning loss (reference-free, length-normalized).

    SimPO uses average log probability as implicit reward,
    with a target reward margin gamma.

    For unlearning (reversed):
    L = -log sigmoid(beta * (avg_log_prob(rejected) - avg_log_prob(chosen)) - gamma)

    Pushes model toward refusal and away from known answers.
    """
    prompt_length = batch["prompt_length"]

    chosen_lp, chosen_mask = _get_per_token_log_probs(
        model, batch["chosen_input_ids"], batch["chosen_attention_mask"], prompt_length
    )
    rejected_lp, rejected_mask = _get_per_token_log_probs(
        model, batch["rejected_input_ids"], batch["rejected_attention_mask"], prompt_length
    )

    # Length-normalized (average) log probs
    avg_chosen = _avg_sequence_log_prob(chosen_lp, chosen_mask)
    avg_rejected = _avg_sequence_log_prob(rejected_lp, rejected_mask)

    # Reversed: prefer rejected (refusal) over chosen (known answer)
    logits = beta * (avg_rejected - avg_chosen) - gamma
    loss = -F.logsigmoid(logits).mean()

    return loss


def fugas_loss(
    model: PreTrainedModel,
    original_model: PreTrainedModel,
    batch: dict[str, Tensor],
    beta: float = 0.1,
) -> Tensor:
    """FUGAS-style: preference optimization against original predictions.

    Uses the model's original (pre-unlearning) predictions as a negative
    reference. Pushes current model outputs to diverge from the original
    on forget data, with bounded loss to prevent gradient explosion.

    L = -beta * KL(current || original) on forget data
      = push current AWAY from original predictions
    """
    prompt_length = batch["prompt_length"]
    input_ids = batch["chosen_input_ids"]
    attention_mask = batch["chosen_attention_mask"]

    # Original model predictions (frozen)
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            orig_outputs = original_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
        orig_logits = orig_outputs.logits[:, prompt_length - 1:-1, :]
        orig_probs = F.softmax(orig_logits, dim=-1)

    # Current model predictions
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        curr_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    curr_logits = curr_outputs.logits[:, prompt_length - 1:-1, :]
    curr_log_probs = F.log_softmax(curr_logits, dim=-1)

    # Response mask
    response_mask = attention_mask[:, prompt_length:].unsqueeze(-1)

    # Reverse KL: push current AWAY from original
    # Standard KL = sum(p * log(p/q)), we negate to maximize divergence
    kl = F.kl_div(curr_log_probs, orig_probs, reduction="none")
    kl = (kl * response_mask).sum(dim=-1).mean()

    # Negate: we want to INCREASE divergence (bounded by sigmoid)
    loss = -F.logsigmoid(beta * kl)

    return loss


# Registry for easy lookup
OBJECTIVE_REGISTRY: dict[str, callable] = {
    "dpo": dpo_unlearn_loss,
    "npo": npo_loss,
    "simnpo": simnpo_loss,
    "simpo": simpo_loss,
    "fugas": fugas_loss,
}

# Which objectives need a reference model
NEEDS_REFERENCE: dict[str, bool] = {
    "dpo": True,
    "npo": True,
    "simnpo": False,
    "simpo": False,
    "fugas": True,  # uses original (pre-unlearning) model as reference
}


def get_unlearn_loss(
    objective: str,
    model: PreTrainedModel,
    batch: dict[str, Tensor],
    ref_model: PreTrainedModel | None = None,
    beta: float = 0.1,
    gamma: float = 1.0,
) -> Tensor:
    """Dispatch to the appropriate unlearning loss function."""
    if objective not in OBJECTIVE_REGISTRY:
        raise ValueError(
            f"Unknown objective '{objective}'. "
            f"Available: {list(OBJECTIVE_REGISTRY.keys())}"
        )

    if NEEDS_REFERENCE[objective] and ref_model is None:
        raise ValueError(f"Objective '{objective}' requires a reference model")

    if objective == "dpo":
        return dpo_unlearn_loss(model, ref_model, batch, beta=beta)
    elif objective == "npo":
        return npo_loss(model, ref_model, batch, beta=beta)
    elif objective == "simnpo":
        return simnpo_loss(model, batch, beta=beta)
    elif objective == "simpo":
        return simpo_loss(model, batch, beta=beta, gamma=gamma)
    elif objective == "fugas":
        return fugas_loss(model, ref_model, batch, beta=beta)
    else:
        raise ValueError(f"Unhandled objective: {objective}")
