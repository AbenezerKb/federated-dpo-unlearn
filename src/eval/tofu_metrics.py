"""TOFU evaluation metrics for federated unlearning."""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def compute_truth_ratio(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_data: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    """Truth Ratio: primary TOFU metric.

    Measures how much the model "knows" the forget data.
    TR = P(correct answer) / (P(correct answer) + P(wrong answer))

    For successful unlearning, TR on forget set should approach 0.5
    (random guessing).
    """
    model.eval()
    truth_ratios = []

    with torch.no_grad():
        for item in eval_data:
            prompt = item["prompt"]
            correct = item["chosen"]
            wrong = item["rejected"]

            correct_log_prob = _compute_sequence_probability(
                model, tokenizer, prompt, correct, device
            )
            wrong_log_prob = _compute_sequence_probability(
                model, tokenizer, prompt, wrong, device
            )

            # Convert log probs to probs for ratio
            correct_prob = np.exp(correct_log_prob)
            wrong_prob = np.exp(wrong_log_prob)

            denom = correct_prob + wrong_prob
            if denom > 0:
                tr = correct_prob / denom
            else:
                tr = 0.5

            truth_ratios.append(tr)

    mean_tr = float(np.mean(truth_ratios))
    return {
        "truth_ratio_mean": mean_tr,
        "truth_ratio_std": float(np.std(truth_ratios)),
        "truth_ratio_median": float(np.median(truth_ratios)),
        "num_samples": len(truth_ratios),
    }


def compute_rouge_l(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_data: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int = 128,
) -> dict[str, float]:
    """ROUGE-L between generated text and ground truth answers."""
    model.eval()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []

    with torch.no_grad():
        for item in eval_data:
            prompt = item["prompt"]
            reference = item["chosen"]

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                    or tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            score = scorer.score(reference, generated)
            scores.append(score["rougeL"].fmeasure)

    return {
        "rouge_l_mean": float(np.mean(scores)),
        "rouge_l_std": float(np.std(scores)),
        "num_samples": len(scores),
    }


def compute_mia_accuracy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    member_data: list[dict[str, Any]],
    nonmember_data: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    """Membership Inference Attack accuracy.

    Uses loss-based MIA: members should have lower loss than non-members.
    For successful unlearning, MIA accuracy on forget set should approach
    50% (can't distinguish members from non-members).
    """
    model.eval()

    member_losses = []
    for item in member_data:
        loss = _compute_sequence_probability(
            model, tokenizer, item["prompt"], item["chosen"], device
        )
        member_losses.append(-loss)  # Negate log prob to get loss

    nonmember_losses = []
    for item in nonmember_data:
        loss = _compute_sequence_probability(
            model, tokenizer, item["prompt"], item["chosen"], device
        )
        nonmember_losses.append(-loss)

    if not member_losses or not nonmember_losses:
        return {"mia_accuracy": 0.5, "auc": 0.5}

    # Find threshold that maximizes accuracy
    all_losses = member_losses + nonmember_losses
    all_labels = [1] * len(member_losses) + [0] * len(nonmember_losses)

    best_acc = 0.5
    thresholds = sorted(set(all_losses))

    for thresh in thresholds:
        preds = [1 if l <= thresh else 0 for l in all_losses]
        acc = sum(
            p == l for p, l in zip(preds, all_labels)
        ) / len(all_labels)
        best_acc = max(best_acc, acc)

    # AUC via simple trapezoidal
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, [-l for l in all_losses])
    except ValueError:
        auc = 0.5

    return {
        "mia_accuracy": best_acc,
        "mia_auc": auc,
        "member_loss_mean": float(np.mean(member_losses)),
        "nonmember_loss_mean": float(np.mean(nonmember_losses)),
    }


def compute_model_utility(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    retain_data: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    """Model utility on retain set.

    Measures how well model preserves knowledge it should keep.
    Higher is better.
    """
    tr_metrics = compute_truth_ratio(model, tokenizer, retain_data, device)
    rouge_metrics = compute_rouge_l(model, tokenizer, retain_data, device)

    return {
        "retain_truth_ratio": tr_metrics["truth_ratio_mean"],
        "retain_rouge_l": rouge_metrics["rouge_l_mean"],
        "model_utility": (
            tr_metrics["truth_ratio_mean"] + rouge_metrics["rouge_l_mean"]
        ) / 2,
    }


def compute_forget_quality(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    forget_data: list[dict[str, Any]],
    retain_data: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, float]:
    """Forget quality: how well model forgot target data.

    Combines Truth Ratio (should be ~0.5) and MIA (should be ~0.5).
    """
    tr_metrics = compute_truth_ratio(model, tokenizer, forget_data, device)

    mia_metrics = compute_mia_accuracy(
        model, tokenizer,
        member_data=forget_data,
        nonmember_data=retain_data[:len(forget_data)],
        device=device,
    )

    # Forget quality: TR close to 0.5 + MIA close to 0.5
    tr_score = 1.0 - abs(tr_metrics["truth_ratio_mean"] - 0.5) * 2
    mia_score = 1.0 - abs(mia_metrics["mia_accuracy"] - 0.5) * 2

    return {
        "forget_truth_ratio": tr_metrics["truth_ratio_mean"],
        "forget_mia_accuracy": mia_metrics["mia_accuracy"],
        "forget_mia_auc": mia_metrics["mia_auc"],
        "forget_quality": (tr_score + mia_score) / 2,
    }


def run_full_evaluation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    forget_data: list[dict[str, Any]],
    retain_data: list[dict[str, Any]],
    device: torch.device,
    objective: str = "",
) -> dict[str, Any]:
    """Run all TOFU metrics and return combined results."""
    logger.info(f"Running full evaluation (objective={objective})")

    results = {"objective": objective}

    # Forget quality
    forget_metrics = compute_forget_quality(
        model, tokenizer, forget_data, retain_data, device
    )
    results.update(forget_metrics)

    # Model utility
    utility_metrics = compute_model_utility(
        model, tokenizer, retain_data, device
    )
    results.update(utility_metrics)

    # Overall score: harmonic mean of forget quality and model utility
    fq = forget_metrics["forget_quality"]
    mu = utility_metrics["model_utility"]
    if fq + mu > 0:
        results["overall_score"] = 2 * fq * mu / (fq + mu)
    else:
        results["overall_score"] = 0.0

    logger.info(
        f"Eval results ({objective}): "
        f"forget_quality={fq:.4f}, "
        f"model_utility={mu:.4f}, "
        f"overall={results['overall_score']:.4f}"
    )

    return results


def _compute_sequence_probability(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> float:
    """Compute log probability of continuation given prompt."""
    full_text = prompt + continuation

    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    full_enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    input_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)
    prompt_len = prompt_enc["input_ids"].shape[1]

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Only continuation tokens
    if prompt_len > 0:
        token_log_probs = token_log_probs[:, prompt_len - 1:]

    total_log_prob = token_log_probs.sum().item()
    return total_log_prob
