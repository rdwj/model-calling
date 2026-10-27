#!/bin/bash

# Set colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run specific tests
run_tests() {
    local test_pattern=$1
    local test_description=$2
    
    echo -e "\n${YELLOW}Running $test_description...${NC}"
    PYTHONPATH=$PWD python -m pytest $test_pattern -v
    echo -e "${GREEN}Completed $test_description.${NC}"
}

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

# Stay in the root directory and use correct paths
if [[ "$1" == "function" || "$1" == "functions" || "$1" == "tool" || "$1" == "tools" ]]; then
    # Run only function calling tests
    run_tests "tests/test_function_calling.py" "function calling tests"
elif [[ "$1" == "basic" ]]; then
    # Run only basic tests
    run_tests "tests/test_basic.py" "basic functionality tests"
elif [[ "$1" == "streaming" || "$1" == "stream" ]]; then
    # Run only streaming tests
    run_tests "tests/test_streaming_and_errors.py::test_streaming*" "streaming tests"
elif [[ "$1" == "errors" || "$1" == "error" ]]; then
    # Run only error tests
    run_tests "tests/test_streaming_and_errors.py::test_invalid_model tests/test_streaming_and_errors.py::test_malformed_request" "error handling tests"
elif [[ "$1" == "quick" || "$1" == "fast" ]]; then
    # Run a subset of tests quickly to verify functionality
    run_tests "tests/test_basic.py::test_service_health" "quick verification tests"
else
    # Run all tests
    echo -e "${YELLOW}Running all tests...${NC}"
    PYTHONPATH=$PWD python -m pytest tests/ -v
    echo -e "${GREEN}Completed all tests.${NC}"
fi

echo -e "\n${YELLOW}Tests complete. Check the output above for any failures.${NC}"
