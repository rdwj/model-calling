#!/bin/bash

# Create an __init__.py file in the tests directory to make it a proper package
echo "# Tests package for model-calling" > tests/__init__.py

# Update the auto_config_tests.sh script to ensure it's creating a proper Python module
if [ -f tests/models_available.py ]; then
    # Add an export line to models_available.py
    echo -e "\n# Export these for other test files to use\n__all__ = ['ALL_MODELS', 'FUNCTION_CALLING_MODELS', 'BASIC_MODELS']" >> tests/models_available.py
    echo "Added export directives to models_available.py"
fi

echo "Fixed Python package structure for tests"
