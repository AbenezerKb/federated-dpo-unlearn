"""Aggregate evaluation results across all objectives."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--objectives", nargs="+", required=True)
    args = parser.parse_args()

    all_results = {}
    for objective in args.objectives:
        results_path = os.path.join(
            args.run_dir, objective, "eval_results.json"
        )
        if os.path.exists(results_path):
            with open(results_path) as f:
                all_results[objective] = json.load(f)
        else:
            print(f"Warning: no results for {objective}")

    if not all_results:
        print("No results found")
        return

    # Print comparison table
    metrics = [
        "forget_truth_ratio",
        "forget_mia_accuracy",
        "forget_quality",
        "retain_truth_ratio",
        "retain_rouge_l",
        "model_utility",
        "overall_score",
    ]

    header = f"{'Metric':<25}" + "".join(
        f"{obj:>12}" for obj in all_results
    )
    print(header)
    print("-" * len(header))

    for metric in metrics:
        row = f"{metric:<25}"
        for obj in all_results:
            val = all_results[obj].get(metric, "N/A")
            if isinstance(val, float):
                row += f"{val:>12.4f}"
            else:
                row += f"{str(val):>12}"
        print(row)

    # Save aggregated
    output_path = os.path.join(args.run_dir, "all_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAggregated results saved: {output_path}")

    # Best objective by overall score
    best = max(
        all_results.items(),
        key=lambda x: x[1].get("overall_score", 0),
    )
    print(f"\nBest objective: {best[0]} (overall={best[1].get('overall_score', 0):.4f})")


if __name__ == "__main__":
    main()
