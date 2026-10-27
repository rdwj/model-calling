#!/bin/bash

# Set colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up tests for Model Calling...${NC}"

# Make all scripts executable
echo -e "Making scripts executable..."
chmod +x make_scripts_executable.sh
./make_scripts_executable.sh

# Create __init__.py file in tests directory if it doesn't exist
if [ ! -f tests/__init__.py ]; then
    echo "# Tests package for model-calling" > tests/__init__.py
    echo -e "${GREEN}Created tests/__init__.py${NC}"
fi

# Auto-detect Ollama models and configure tests
echo -e "\n${YELLOW}Auto-detecting Ollama models...${NC}"
./auto_config_tests.sh

# Run a quick verification test
echo -e "\n${YELLOW}Running a quick verification test...${NC}"
./test_ollama.sh quick

echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "You can now run tests with:"
echo -e "${YELLOW}./test_ollama.sh${NC} - Run all tests"
echo -e "${YELLOW}./test_ollama.sh function${NC} - Run function calling tests"
echo -e "${YELLOW}./test_ollama.sh basic${NC} - Run basic tests"
