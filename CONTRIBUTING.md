# Contributing to Model Calling

Thank you for your interest in contributing to the Model Calling project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Pull Request Process](#pull-request-process)
- [Adding New Model Adapters](#adding-new-model-adapters)
- [Testing Guidelines](#testing-guidelines)
- [Style Guidelines](#style-guidelines)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. Please be considerate in your interactions with others.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
   ```bash
   git clone https://github.com/<your-username>/model-calling.git
   cd model-calling
   ```
3. **Set up the development environment**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e '.[dev]'
   ```
4. **Create a new branch** for your feature or bugfix
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Process

1. **Choose an issue** to work on, or create a new one if you've identified a bug or improvement.
2. **Comment on the issue** to let others know you're working on it.
3. **Implement your changes** following the project's style and testing guidelines.
4. **Write or update tests** for your changes.
5. **Ensure all tests pass** by running the test scripts:
   ```bash
   ./run_tests.sh
   ```
6. **Update documentation** as needed.

## Pull Request Process

1. **Push your changes** to your fork on GitHub.
2. **Create a Pull Request** against the main repository's `main` branch.
3. **Describe your changes** in detail, referencing any related issues.
4. **Wait for a review** from the maintainers.
5. **Address any feedback** received during the review process.

## Adding New Model Adapters

One of the most valuable contributions is adding support for new model backends. To add a new adapter:

1. Create a new adapter class that extends `ModelAdapter` (and `StreamingAdapter` if appropriate).
2. Implement all required methods for translating between the OpenAI API format and your model's format.
3. Register the adapter with the registry in an appropriate place.
4. Add capability detection for the new model type.
5. Include tests that verify basic functionality, function calling, and streaming (if supported).

Example:

```python
from model_calling.adapter import ModelAdapter
from model_calling.registry import registry

class MyCustomAdapter(ModelAdapter):
    def __init__(self, model_config):
        super().__init__(model_config)
        # Initialize model-specific config
        
    async def translate_request(self, openai_format_request):
        # Convert from OpenAI format to your model's format
        
    async def translate_response(self, model_response):
        # Convert from your model's response to OpenAI format
        
    async def call_model(self, translated_request):
        # Make the API call to your model

# Register in appropriate initialization code
registry.register_adapter_class("mymodel", MyCustomAdapter)
```

## Testing Guidelines

- **Write tests** for all new functionality.
- **Run the full test suite** before submitting pull requests.
- **Test edge cases** and error handling.
- The project includes several test scripts:
  - `run_tests.sh`: Run all tests
  - `test_ollama.sh`: Test Ollama adapter specifically
  - `diagnose_models.sh`: Check model compatibility

## Style Guidelines

- Follow **PEP 8** coding style guidelines.
- Use **meaningful variable names** that clearly describe their purpose.
- Write **clear comments and docstrings** for functions and classes.
- Be **consistent with existing code** style and patterns.

## Documentation

- Update the **README.md** when adding significant features.
- Document **all public methods** with clear docstrings.
- Add **examples** for new functionality.

## Issue Reporting

When reporting issues, please include:

- A **clear description** of the problem
- **Steps to reproduce** the issue
- **Expected behavior** versus actual behavior
- **Environment details** (OS, Python version, dependencies)
- **Logs** or error messages if available

## Feature Requests

For feature requests, please describe:

- The **specific functionality** you'd like to see
- **Use cases** for the feature
- Any **implementation ideas** you may have

---

Thank you for contributing to Model Calling! Your efforts help improve the project for everyone.