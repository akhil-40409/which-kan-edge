# Contributing Guidelines

Thank you for considering contributing to this research project! This document outlines the collaborative coding standards and workflow for contributing to this repository.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Verification Requirements](#verification-requirements)
- [Coding Standards](#coding-standards)
- [Commit Message Guidelines](#commit-message-guidelines)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing viewpoints and experiences
- Accept responsibility and apologize for mistakes

## Getting Started

1. **Fork the repository** and clone your fork locally
2. **Set up your development environment**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   pre-commit install
   ```
3. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Before Making Changes

1. Ensure you're working on the latest code:
   ```bash
   git checkout main
   git pull origin main
   ```

2. Create a new branch from main:
   ```bash
   git checkout -b feature/descriptive-name
   ```

### While Making Changes

1. **Write clean, documented code** following our coding standards
2. **Add type hints** to all function signatures
3. **Write or update tests** for your changes
4. **Update documentation** if you're changing functionality
5. **Run tests frequently** to catch issues early:
   ```bash
   pytest tests/
   ```

### Before Committing

Pre-commit hooks will automatically run, but you can also run checks manually:

```bash
# Run all pre-commit checks
pre-commit run --all-files

# Run specific checks
black src experiments tests
isort src experiments tests
flake8 --config .dev-config/.flake8 src experiments tests
mypy --config-file .dev-config/mypy.ini src experiments
pylint --rcfile .dev-config/.pylintrc src experiments
pytest tests/
```

## Pull Request Process

### Creating a Pull Request

1. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request** on GitHub with a clear title and description

3. **Fill out the PR template** completely (see below)

### Pull Request Template

Every pull request **MUST** include the following sections:

```markdown
## Description

[Provide a clear and concise description of your changes]

## Motivation and Context

[Why is this change required? What problem does it solve?]

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Experiment/research code

## Related Issues

[Link to related issues, e.g., "Closes #123" or "Related to #456"]

## Verification

**This section is MANDATORY for all PRs involving code changes.**

### Testing Performed

[Describe what testing you performed]

Example:
- Unit tests added/updated: [list test files]
- Integration tests run: [describe]
- Manual testing performed: [describe]

### Sample Data Runs

**For research/experiment code, you MUST provide evidence of runs on sample data.**

Provide:
1. **Command used to run**:
   ```bash
   python experiments/my_experiment.py --config example
   ```

2. **Sample output/results**:
   ```
   [Paste relevant output, metrics, or logs]
   ```

3. **Artifacts** (if applicable):
   - Attach screenshots of plots/visualizations
   - Link to W&B run: [wandb-run-url]
   - Configuration used: [link to config file or paste inline]

4. **Performance metrics** (if applicable):
   - Runtime: X seconds/minutes
   - Memory usage: Y GB
   - Accuracy/Loss: [relevant metrics]

### Environment

- Python version: [e.g., 3.9]
- Operating System: [e.g., Ubuntu 22.04, macOS 13]
- Key dependencies: [if relevant to the change]

## Checklist

Before submitting your PR, ensure:

- [ ] My code follows the project's coding standards
- [ ] I have added type hints to all function signatures
- [ ] I have written/updated tests that prove my fix/feature works
- [ ] All tests pass locally (`pytest tests/`)
- [ ] I have updated the documentation accordingly
- [ ] My changes generate no new warnings from linters
- [ ] I have added docstrings to all new functions/classes (Google style)
- [ ] I have included the Verification section with sample data runs (if applicable)
- [ ] I have run the code on sample/test data and verified it works
- [ ] Pre-commit hooks pass without errors
```

### Review Process

1. **Automated checks** will run via GitHub Actions
2. **At least one reviewer** must approve your PR
3. **Address all comments** from reviewers promptly
4. **Update your PR** based on feedback

## Verification Requirements

### For All Code Changes

- All tests must pass
- Code coverage should not decrease
- All linters must pass without errors
- Pre-commit hooks must succeed

### For Research/Experiment Code

**MANDATORY: You must demonstrate that your code runs successfully on sample data.**

This requirement ensures:
- The code is functional and not just syntactically correct
- Results are reproducible
- Other researchers can validate your approach
- The implementation matches the intended algorithm/method

#### What to Include

1. **Minimal working example**: Show the simplest way to run your code
2. **Sample dataset**: Use a small, representative dataset (can be synthetic)
3. **Expected behavior**: Document what the output should look like
4. **Screenshots/plots**: Visual evidence of results (if applicable)
5. **Metrics**: Quantitative results or validation metrics

#### Example Verification Section

```markdown
## Verification

### Testing Performed

Added unit tests in `tests/test_new_model.py` covering:
- Model initialization
- Forward pass
- Loss calculation
- Parameter updates

All tests pass:
```bash
$ pytest tests/test_new_model.py -v
tests/test_new_model.py::test_model_init PASSED
tests/test_new_model.py::test_forward_pass PASSED
tests/test_new_model.py::test_loss_calculation PASSED
tests/test_new_model.py::test_parameter_update PASSED
```

### Sample Data Runs

Ran the new model on synthetic MNIST-like data (100 samples):

**Command**:
```bash
python experiments/train_new_model.py data.num_samples=100 training.epochs=5
```

**Output**:
```
Epoch 1/5: Loss=0.892, Accuracy=0.645
Epoch 2/5: Loss=0.543, Accuracy=0.782
Epoch 3/5: Loss=0.321, Accuracy=0.891
Epoch 4/5: Loss=0.198, Accuracy=0.934
Epoch 5/5: Loss=0.145, Accuracy=0.956
```

**W&B Run**: https://wandb.ai/project/runs/abc123

**Configuration Used**:
```yaml
model:
  name: "new_model_v1"
  hidden_size: 128
  dropout: 0.1

training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 5
```

**Performance**:
- Training time: 2.3 minutes
- Memory usage: ~500MB
- Final validation accuracy: 95.6%

### Environment

- Python 3.10
- Ubuntu 22.04
- PyTorch 2.0.1
- CUDA 11.8
```

## Coding Standards

### Python Style

- **Line length**: Maximum 100 characters
- **Formatting**: Use `black` (automatically enforced)
- **Import sorting**: Use `isort` (automatically enforced)
- **Style guide**: Follow PEP 8 (enforced by `flake8`)

### Type Hints

All functions should include type hints:

```python
from typing import List, Optional, Dict, Any

def process_data(
    data: List[float],
    config: Dict[str, Any],
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """Process the input data according to configuration.
    
    Args:
        data: List of numerical values to process.
        config: Configuration dictionary with processing parameters.
        threshold: Optional threshold value for filtering.
    
    Returns:
        Dictionary containing processed results and metadata.
    """
    # Implementation here
    pass
```

### Documentation

Use Google-style docstrings for all public functions and classes:

```python
def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics for predictions.
    
    This function computes various metrics including accuracy, precision,
    recall, and F1 score for the given predictions and targets.
    
    Args:
        predictions: Array of predicted values with shape (n_samples,).
        targets: Array of ground truth values with shape (n_samples,).
    
    Returns:
        Dictionary containing metric names as keys and computed values.
        Keys include: 'accuracy', 'precision', 'recall', 'f1_score'.
    
    Raises:
        ValueError: If predictions and targets have different shapes.
        
    Example:
        >>> preds = np.array([0, 1, 1, 0])
        >>> targets = np.array([0, 1, 0, 0])
        >>> metrics = calculate_metrics(preds, targets)
        >>> print(metrics['accuracy'])
        0.75
    """
    # Implementation here
    pass
```

### Testing

- Write tests for all new functionality
- Aim for >80% code coverage
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern

```python
def test_calculate_metrics_with_perfect_predictions():
    """Test metric calculation when all predictions are correct."""
    # Arrange
    predictions = np.array([0, 1, 1, 0])
    targets = np.array([0, 1, 1, 0])
    
    # Act
    metrics = calculate_metrics(predictions, targets)
    
    # Assert
    assert metrics['accuracy'] == 1.0
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
```

### Experiment Code

For research experiments:

- Use Hydra for configuration management
- Log all experiments to W&B
- Include configuration files in `experiments/configs/`
- Document hyperparameters and their expected ranges
- Save model checkpoints and artifacts
- Log random seeds for reproducibility

## Commit Message Guidelines

Write clear, concise commit messages:

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, missing semicolons, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

```
feat(experiments): add new attention mechanism for transformer

Implemented multi-head attention with configurable number of heads
and dropout. Includes unit tests and sample experiment.

Related to #42
```

```
fix(data): correct normalization in preprocessing pipeline

The normalization was using incorrect mean/std values. Updated to
use per-channel statistics for image data.

Fixes #38
```

## Questions or Issues?

If you have questions about contributing:

1. Check existing documentation
2. Search closed issues and PRs
3. Open a new issue with the "question" label
4. Reach out to maintainers

## Recognition

All contributors will be acknowledged in the project documentation. Thank you for helping improve this research project!
