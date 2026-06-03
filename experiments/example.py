"""Example experiment script demonstrating Hydra and W&B integration.

This script shows how to set up a basic experiment with configuration
management using Hydra and experiment tracking using Weights & Biases.
"""

from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf


def run_experiment(config: Dict[str, Any]) -> Dict[str, float]:
    """Run a simple experiment with the given configuration.

    Args:
        config: Configuration dictionary containing experiment parameters.

    Returns:
        Dictionary with experiment results.
    """
    print(f"Running experiment with config: {config}")

    # Simulate experiment
    learning_rate = config.get("learning_rate", 0.001)
    epochs = config.get("epochs", 10)

    results = {
        "final_loss": 0.123,
        "final_accuracy": 0.95,
        "learning_rate": learning_rate,
        "epochs": epochs,
    }

    print(f"Results: {results}")
    return results


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main experiment entry point.

    Args:
        cfg: Hydra configuration object.
    """
    print("=" * 60)
    print("Starting Example Experiment")
    print("=" * 60)

    # Convert config to dictionary
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config_dict, dict):
        raise TypeError("Configuration must be a dictionary")
    print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Uncomment to enable W&B tracking:
    # import wandb
    # wandb.init(
    #     project="research-project",
    #     config=config_dict,
    #     name="example-experiment"
    # )

    # Run experiment
    _ = run_experiment(config_dict)  # type: ignore[arg-type]

    # Log results to W&B (uncomment when using W&B):
    # wandb.log(results)
    # wandb.finish()

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
