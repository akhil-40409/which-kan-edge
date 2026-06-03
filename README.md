# Research Project Template

A comprehensive Python project template designed specifically for research development (not production). This template provides a structured, reproducible environment for conducting computational research experiments.

## Features

- **Modern Package Management**: Uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management
- **Structured Directory Layout**: Organized separation of source code, experiments, and tests
- **Comprehensive Linting & Type Checking**: Pre-configured with mypy, isort, pylint, black, pydocstyle, and flake8
- **Pre-commit Hooks**: Automated code quality checks before each commit
- **CI/CD Integration**: GitHub Actions workflow for continuous integration on PRs
- **Experiment Tracking**: Ready-to-use configuration for Hydra and Weights & Biases (W&B) (optional dependencies)

## Directory Structure

```
.
├── src/                    # Source code for your research project
│   └── __init__.py
├── experiments/            # Research experiment scripts and notebooks
│   ├── configs/           # Hydra configuration files
│   └── example.py         # Example experiment script
├── tests/                  # Pytest test suite
│   └── test_example.py
├── .dev-config/           # Development tool configurations
│   ├── mypy.ini
│   ├── .flake8
│   ├── .pylintrc
│   └── .pydocstyle
├── .github/
│   └── workflows/
│       └── pr-monitor.yml  # CI/CD pipeline
├── pyproject.toml          # Project dependencies and tool configurations
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── README.md               # This file
└── CONTRIBUTING.md         # Contribution guidelines
```

## Getting Started

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installing uv

Install uv using the official installer:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setting Up the Development Environment

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create a virtual environment and install dependencies**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Running Experiments

This template is designed to support reproducible research experiments using [Hydra](https://hydra.cc/) for configuration management and [Weights & Biases (W&B)](https://wandb.ai/) for experiment tracking.

> **Note**: Hydra and W&B are included as dependencies in this template for convenience, but they are **optional**. You can remove them from `pyproject.toml` if you don't need experiment tracking or configuration management features.

#### Basic Experiment Execution

Place your experiment scripts in the `experiments/` directory:

```bash
python experiments/example.py
```

#### Using Hydra for Configuration Management

Hydra allows you to manage experiment configurations through YAML files and command-line overrides.

1. **Create a configuration directory**:
   ```bash
   mkdir -p experiments/configs
   ```

2. **Create a configuration file** (`experiments/configs/config.yaml`):
   ```yaml
   # Example configuration
   model:
     name: "resnet50"
     learning_rate: 0.001
     batch_size: 32
   
   data:
     dataset: "imagenet"
     num_workers: 4
   
   training:
     epochs: 100
     device: "cuda"
   ```

3. **Use Hydra in your experiment script**:
   ```python
   import hydra
   from omegaconf import DictConfig

   @hydra.main(version_base=None, config_path="configs", config_name="config")
   def main(cfg: DictConfig) -> None:
       print(f"Model: {cfg.model.name}")
       print(f"Learning rate: {cfg.model.learning_rate}")
       # Your experiment code here

   if __name__ == "__main__":
       main()
   ```

4. **Run experiments with different configurations**:
   ```bash
   # Use default config
   python experiments/my_experiment.py

   # Override specific parameters
   python experiments/my_experiment.py model.learning_rate=0.01 training.epochs=50

   # Use a different config file
   python experiments/my_experiment.py --config-name=config_alt
   ```

#### Using Weights & Biases for Experiment Tracking

W&B provides comprehensive experiment tracking, visualization, and collaboration features.

1. **Set up W&B**:
   ```bash
   wandb login
   ```

2. **Initialize W&B in your experiment**:
   ```python
   import wandb

   # Initialize a W&B run
   wandb.init(
       project="my-research-project",
       config={
           "learning_rate": 0.001,
           "epochs": 100,
           "batch_size": 32,
       }
   )

   # Log metrics during training
   for epoch in range(epochs):
       # ... training code ...
       wandb.log({
           "epoch": epoch,
           "loss": loss,
           "accuracy": accuracy,
       })

   # Finish the run
   wandb.finish()
   ```

3. **Integration with Hydra**:
   ```python
   import hydra
   import wandb
   from omegaconf import DictConfig, OmegaConf

   @hydra.main(version_base=None, config_path="configs", config_name="config")
   def main(cfg: DictConfig) -> None:
       # Convert Hydra config to dict for W&B
       wandb.init(
           project="my-research-project",
           config=OmegaConf.to_container(cfg, resolve=True)
       )
       
       # Your experiment code here
       
       wandb.finish()

   if __name__ == "__main__":
       main()
   ```

## Development Workflow

### Code Quality Tools

This template uses several tools to maintain code quality:

- **black**: Code formatter (line length: 100)
- **isort**: Import statement sorter
- **flake8**: Style guide enforcement
- **pydocstyle**: Docstring style checker
- **mypy**: Static type checker
- **pylint**: Comprehensive code analyzer

### Running Quality Checks Manually

```bash
# Format code
black src experiments tests

# Sort imports
isort src experiments tests

# Check style
flake8 --config .dev-config/.flake8 src experiments tests

# Check docstrings
pydocstyle --config .dev-config/.pydocstyle src experiments

# Type checking
mypy --config-file .dev-config/mypy.ini src experiments

# Lint code
pylint --rcfile .dev-config/.pylintrc src experiments
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_example.py

# Run tests matching a pattern
pytest -k "test_function_name"
```

### Pre-commit Hooks

Pre-commit hooks automatically run quality checks before each commit. They will:
- Format code with black
- Sort imports with isort
- Check code style with flake8
- Validate docstrings with pydocstyle
- Perform type checking with mypy
- Run pylint analysis
- Execute the test suite

To run pre-commit checks manually:
```bash
pre-commit run --all-files
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/pr-monitor.yml`) automatically runs on every pull request and push to main/develop branches. It:

1. Tests against multiple Python versions (3.11, 3.12)
2. Runs all code quality checks (black, isort, flake8, pydocstyle, mypy, pylint)
3. Executes the test suite with coverage reporting
4. Uploads coverage reports to Codecov

## Adding New Dependencies

Use uv to add new dependencies to your project. The `uv add` command automatically installs the package and updates `pyproject.toml`:

```bash
# Add a runtime dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Add a specific version
uv add "package-name>=1.0.0"

# Add multiple packages at once
uv add package1 package2 package3
```

After adding dependencies, make sure to commit the updated `pyproject.toml` to keep your project configuration in sync.

## Configuration Files

All linter and type-checker configurations are located in the `.dev-config/` directory:

- **`.dev-config/mypy.ini`**: mypy type checking configuration
- **`.dev-config/.flake8`**: flake8 style checking configuration
- **`.dev-config/.pylintrc`**: pylint linting configuration
- **`.dev-config/.pydocstyle`**: pydocstyle docstring checking configuration

Tool configurations in `pyproject.toml`:
- **`[tool.black]`**: Black formatter settings
- **`[tool.isort]`**: isort import sorting settings
- **`[tool.pytest.ini_options]`**: pytest test runner settings

## Best Practices

1. **Keep experiments reproducible**: Use Hydra configs and log random seeds
2. **Track all experiments**: Initialize W&B for every experiment run
3. **Write tests**: Add tests for all non-trivial functions in `src/`
4. **Document your code**: Follow Google-style docstrings
5. **Use type hints**: Add type annotations to function signatures
6. **Commit often**: Make small, focused commits with clear messages
7. **Review before pushing**: Let pre-commit hooks catch issues early

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues, please open an issue on the GitHub repository.
