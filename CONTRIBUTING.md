# Contributing to GPU Memory Calculator

Thank you for considering contributing to GPU Memory Calculator! This document provides guidelines and instructions for contributing.

## 🎯 Ways to Contribute

- 🐛 Report bugs and issues
- 💡 Suggest new features or enhancements
- 📝 Improve documentation
- 🔧 Submit bug fixes or new features
- ✅ Add or improve tests
- 🎨 Improve UI/UX of the web interface

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git for version control
- Familiarity with GPU memory concepts (helpful but not required)

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/gpu-mem-calculator.git
   cd gpu-mem-calculator
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev,web]"
   ```

4. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🔨 Development Workflow

### Code Style

We use automated tools to maintain code quality:

- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **MyPy** for type checking

Format your code before committing:
```bash
black src/ cli/ web/ tests/
ruff check src/ cli/ web/ tests/
mypy src/
```

### Testing

Run the test suite to ensure your changes don't break existing functionality:

```bash
pytest tests/
```

For coverage reports:
```bash
pytest --cov=gpu_mem_calculator tests/
```

### Commit Messages

Write clear, descriptive commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Keep the first line under 72 characters
- Reference issues and pull requests when applicable

Examples:
```
Add support for Mixtral MoE model preset
Fix memory calculation for FSDP with activation checkpointing
Update documentation for DeepSpeed ZeRO-3 configuration
```

## 📋 Pull Request Process

1. **Update tests**: Add tests for new features or bug fixes
2. **Update documentation**: Update README.md and docstrings as needed
3. **Run the test suite**: Ensure all tests pass
4. **Format your code**: Run Black, Ruff, and MyPy
5. **Create a pull request**: Provide a clear description of your changes

### Pull Request Guidelines

- Fill out the PR template completely
- Link related issues (e.g., "Fixes #123")
- Keep PRs focused on a single feature or fix
- Ensure CI checks pass
- Be responsive to feedback and questions

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. **Description**: A clear description of the bug
2. **Steps to reproduce**: Detailed steps to reproduce the issue
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: Python version, OS, GPU type (if relevant)
6. **Configuration**: Include relevant configuration files or command-line arguments

## 💡 Suggesting Features

We welcome feature suggestions! When suggesting a feature:

1. Check if the feature has already been suggested
2. Clearly describe the feature and its use case
3. Explain why this feature would be valuable
4. Consider implementation complexity and maintenance burden

## 📝 Documentation

Good documentation is crucial. When contributing documentation:

- Use clear, concise language
- Include examples where appropriate
- Update both README.md and inline docstrings
- Test all code examples

## 🎨 Adding Model Presets

To add a new model preset:

1. Add the preset configuration to `src/gpu_mem_calculator/presets/models.py`
2. Follow the existing pattern for preset definitions
3. Include accurate model parameters (verify from official sources)
4. Add documentation in the README.md
5. Test the preset with the CLI and web interface

Example:
```python
"llama3-8b": ModelPreset(
    name="LLaMA 3 8B",
    num_parameters=8_000_000_000,
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    vocab_size=32000,
    max_seq_len=8192,
    description="Meta's LLaMA 3 8B parameter model",
),
```

## 🔬 Adding Training Engine Support

To add support for a new training engine:

1. Create a new calculator class in `src/gpu_mem_calculator/core/engines/`
2. Implement the required interface methods
3. Add comprehensive tests
4. Document the memory formulas and sources
5. Update the README with engine-specific information

## ⚖️ Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code. Please report unacceptable behavior to the project maintainers.

## 📜 License

By contributing to GPU Memory Calculator, you agree that your contributions will be licensed under the MIT License.

## ❓ Questions?

If you have questions about contributing, feel free to:

- Open an issue with the "question" label
- Reach out to the maintainers
- Check existing issues and discussions

## 🙏 Thank You!

Your contributions help make GPU Memory Calculator better for everyone. Whether you're fixing a typo, adding a feature, or improving documentation, every contribution is valuable!
