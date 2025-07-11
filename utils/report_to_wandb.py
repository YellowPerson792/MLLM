import json
import argparse
import wandb

def main():
    parser = argparse.ArgumentParser(description="Visualize Trainer metrics in W&B under 'train' namespace")
    parser.add_argument("--trainer_state", type=str, required=True,
                        help="Path to trainer_state.json")
    parser.add_argument("--wandb_project", type=str, required=True,
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, required=True,
                        help="W&B run name for logging metrics")
    args = parser.parse_args()

    # Load trainer state
    with open(args.trainer_state, "r") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    if not log_history:
        print("No log_history found in trainer_state.json")
        return

    # Initialize a new W&B run
    wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    # Define metrics under 'train' group
    wandb.define_metric("train/step")
    wandb.define_metric("train/epoch")
    # infer other metrics keys dynamically below

    for entry in log_history:
        step = entry.get("step")
        epoch = entry.get("epoch")
        # Collect relevant metrics
        for key, value in entry.items():
            if key in ["step", "epoch", "total_flos", "train_runtime", "train_samples_per_second", "train_steps_per_second"]:
                continue
            # prefix with 'train/'
            metric_name = f"train/{key}"
            wandb.define_metric(metric_name, step_metric="train/step")
        # Prepare log dict
        log_dict = {f"train/{k}": v for k,v in entry.items() if k not in ["total_flos", "train_runtime", "train_samples_per_second", "train_steps_per_second"]}
        wandb.log(log_dict)

    print("Metrics logged under 'train' namespace in W&B.")
    wandb.finish()

if __name__ == "__main__":
    main()
