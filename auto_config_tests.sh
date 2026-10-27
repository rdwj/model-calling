#!/bin/bash

# Set colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

MODELS_FILE="tests/models_available.py"

echo -e "${YELLOW}Auto-detecting Ollama models for testing...${NC}"

# Check if ollama command is available
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}Ollama command not found!${NC}"
    echo -e "Please make sure Ollama is installed and in your PATH."
    exit 1
fi

# Run ollama list and capture output
OLLAMA_LIST=$(ollama list)
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to run 'ollama list'${NC}"
    echo "Error: $OLLAMA_LIST"
    exit 1
fi

# Extract model names
MODEL_NAMES=$(echo "$OLLAMA_LIST" | grep -v "^NAME" | awk '{print $1}')
if [ -z "$MODEL_NAMES" ]; then
    echo -e "${RED}No models found!${NC}"
    echo "Please pull some models with 'ollama pull [model]'"
    exit 1
fi

# Count models
MODEL_COUNT=$(echo "$MODEL_NAMES" | wc -l | tr -d ' ')
echo -e "${GREEN}Found $MODEL_COUNT models installed in Ollama${NC}"

# Generate Python lists of models
ALL_MODELS_LIST="ALL_MODELS = ["
for model in $MODEL_NAMES; do
    ALL_MODELS_LIST+="\n    \"$model\","
done
ALL_MODELS_LIST+="\n]"

# Generate categorized model lists
FUNCTION_CALLING_MODELS=$(echo "$MODEL_NAMES" | grep -i -E "mistral|llama|claude|gpt|gemma|granite")
if [ -z "$FUNCTION_CALLING_MODELS" ]; then
    # If no models matched our patterns, use the first model as a fallback
    FUNCTION_CALLING_MODELS=$(echo "$MODEL_NAMES" | head -1)
fi

FUNCTION_CALLING_LIST="FUNCTION_CALLING_MODELS = ["
for model in $FUNCTION_CALLING_MODELS; do
    FUNCTION_CALLING_LIST+="\n    \"$model\","
done
FUNCTION_CALLING_LIST+="\n]"

# All models can be used for basic testing
BASIC_MODELS_LIST="BASIC_MODELS = ["
for model in $MODEL_NAMES; do
    BASIC_MODELS_LIST+="\n    \"$model\","
done
BASIC_MODELS_LIST+="\n]"

# Create the models_available.py file
echo -e "Creating $MODELS_FILE..."
cat > "$MODELS_FILE" <<EOF
"""
Auto-generated file with available Ollama models.
Last updated: $(date)
"""

# All available models
$ALL_MODELS_LIST

# Models likely to support function calling
$FUNCTION_CALLING_LIST

# Models for basic testing
$BASIC_MODELS_LIST

# Export these for other test files to use
__all__ = ['ALL_MODELS', 'FUNCTION_CALLING_MODELS', 'BASIC_MODELS']
EOF

echo -e "${GREEN}Models file created successfully at $MODELS_FILE${NC}"
echo -e "Using the following models for testing:"

# Show first two function calling models
echo -e "${YELLOW}Function calling tests:${NC}"
echo "$FUNCTION_CALLING_MODELS" | head -2 | sed 's/^/- /'

# Show first two basic models
echo -e "${YELLOW}Basic tests:${NC}"
echo "$MODEL_NAMES" | head -2 | sed 's/^/- /'

echo -e "\n${GREEN}Auto-configuration complete!${NC}"
echo -e "The tests will now use your locally available models."
echo -e "You can now run ${YELLOW}./test_ollama.sh quick${NC} to verify."
