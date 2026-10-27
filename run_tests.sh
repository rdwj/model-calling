#!/bin/bash

# Set colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Checking if model-calling service is running...${NC}"

# Check if the service is running
if ! curl -s http://localhost:8000 > /dev/null; then
    echo -e "${RED}The model-calling service doesn't appear to be running!${NC}"
    echo -e "${YELLOW}Start it with:${NC} ./start_service.sh"
    echo -e "${YELLOW}or:${NC} python -m model_calling.server"
    exit 1
fi

echo -e "${GREEN}Model-calling service is running.${NC}"
echo -e "${YELLOW}Running tests...${NC}"

# Stay in the root directory for correct imports
PYTHONPATH=$PWD python -m pytest tests/ "$@" --verbose

echo -e "\n${YELLOW}Tests complete. Check the output above for any failures.${NC}"
