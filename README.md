# Federated DPO Unlearning

Federated unlearning for preference-aligned LLMs using client-level DPO, NPO, SimNPO, SimPO, and FUGAS-style objectives. Evaluated on the TOFU benchmark with Flower-based FL simulation.

## Research Question

Can a client's preference contribution be removed from a federally DPO-aligned LLM such that the model behaves as if that client's (chosen, rejected) pairs never influenced alignment?

## Setup

```bash
# Create environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Login to HuggingFace (for Llama 3.1 access)
huggingface-cli login
```

### Hardware Requirements

- **GPU 0**: Training GPU (recommended: 48GB+ VRAM) — holds model + LoRA + optimizer
- **GPU 1**: Reference model + evaluation (recommended: 24GB+ VRAM)

Tested on: RTX PRO 6000 (96GB) + RTX 4090 (24GB).

## Run Experiments

```bash
# Run all 5 objectives
bash scripts/run_experiment.sh

# Run specific objective
bash scripts/run_experiment.sh configs/default.yaml dpo

# Run multiple specific objectives
bash scripts/run_experiment.sh configs/default.yaml "dpo,npo,simnpo"

# Custom output directory
bash scripts/run_experiment.sh configs/default.yaml all outputs/my_run
```

### Individual Phases

```bash
# Phase 1: Alignment only
python3 -m scripts.run_phase --config configs/default.yaml --phase alignment --output-dir outputs/alignment

# Phase 2: Unlearning
python3 -m scripts.run_phase --config configs/default.yaml --phase unlearning --objective simnpo --alignment-checkpoint outputs/alignment --output-dir outputs/simnpo

# Phase 3: Evaluation
python3 -m scripts.run_phase --config configs/default.yaml --phase eval --objective simnpo --model-checkpoint outputs/simnpo/unlearned_model --output-dir outputs/simnpo
```

## Unlearning Objectives

| Objective | Reference Model | Description |
|-----------|:--------------:|-------------|
| **DPO** | Yes | Reversed DPO — swap chosen/rejected to push model away from known answers |
| **NPO** | Yes | Bounded gradient ascent on forget data relative to reference |
| **SimNPO** | No | Reference-free, pushes average log-prob of forget tokens down |
| **SimPO** | No | Reference-free, length-normalized, with target reward margin |
| **FUGAS** | Yes | Original predictions as negative reference, reverse KL divergence |

## Evaluation Metrics (TOFU)

- **Truth Ratio**: P(correct) / (P(correct) + P(wrong)) — should approach 0.5 for forget set
- **ROUGE-L**: Generation quality on retain set — should stay high
- **MIA Accuracy**: Membership inference attack — should approach 0.5 for forget set
- **Model Utility**: Combined retain performance (higher is better)
- **Forget Quality**: Combined forget metrics (higher means better unlearning)
- **Overall Score**: Harmonic mean of forget quality and model utility

## Configuration

Edit `configs/default.yaml` to modify:
- Model (name, LoRA rank, max length)
- FL settings (num clients, rounds, batch size)
- Unlearning hyperparameters (beta, gamma, learning rate)
- Gradient shielding (enable/disable, subspace dim)
- Data (TOFU split, forget ratio, IID vs non-IID)
- Hardware (GPU assignments)

## Project Structure

```
configs/default.yaml          # Experiment configuration
scripts/
  run_experiment.sh           # Main runner (all 3 phases, loops objectives)
  run_phase.py                # Single phase runner
  aggregate_results.py        # Compare results across objectives
src/
  data/tofu_loader.py         # TOFU loading, federated partitioning, preference pairs
  eval/tofu_metrics.py        # Truth Ratio, ROUGE-L, MIA, forget quality, model utility
  fl/
    flower_client.py          # Flower NumPyClient for alignment + unlearning
    flower_server.py          # FedAvg + gradient shielding strategy
    gradient_shielding.py     # FUGAS-style SVD projection
  models/model_loader.py      # Llama + LoRA loading, checkpointing
  unlearn/objectives.py       # All 5 unlearning loss functions
  utils/gpu.py                # GPU device assignment, memory monitoring
```
