#!/bin/bash

# Set colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Diagnosing Model Calling service...${NC}"

# Check if the service is running
if ! curl -s http://localhost:8000 > /dev/null; then
    echo -e "${RED}The model-calling service doesn't appear to be running!${NC}"
    echo -e "${YELLOW}Start it with:${NC} ./start_service.sh"
    echo -e "${YELLOW}or:${NC} python -m model_calling.server"
    exit 1
fi

echo -e "${GREEN}Model-calling service is running.${NC}"

# Check service status
echo -e "\n${YELLOW}Checking service endpoints...${NC}"
MODELS_RESPONSE=$(curl -s http://localhost:8000/v1/models)
if [[ "$MODELS_RESPONSE" == *"data"* ]]; then
    echo -e "${GREEN}Models endpoint working!${NC}"
    # Extract model count
    MODEL_COUNT=$(echo "$MODELS_RESPONSE" | grep -o '"id"' | wc -l | xargs)
    echo -e "Found ${GREEN}$MODEL_COUNT${NC} models available."
else
    echo -e "${RED}Models endpoint not working correctly.${NC}"
    echo "Response: $MODELS_RESPONSE"
fi

# Test models that should work
echo -e "\n${YELLOW}Testing model compatibility...${NC}"
declare -a TEST_MODELS=("mistral:latest" "mistral-small3.1:24b" "llama3:latest")

for model in "${TEST_MODELS[@]}"; do
    echo -e "\nTesting ${YELLOW}$model${NC}..."
    
    RESPONSE=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"ollama/$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}" \
        -w "%{http_code}")
    
    HTTP_CODE=${RESPONSE: -3}
    RESPONSE_BODY=${RESPONSE:0:${#RESPONSE}-3}
    
    if [[ "$HTTP_CODE" == "200" ]]; then
        echo -e "${GREEN}Success!${NC} Model $model is working correctly."
    else
        echo -e "${RED}Failed!${NC} Model $model returned HTTP code $HTTP_CODE"
        echo "Response body:"
        echo "$RESPONSE_BODY" | head -20
    fi
done

# Test function calling if possible
echo -e "\n${YELLOW}Testing function calling with mistral-small3.1:24b...${NC}"
FC_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ollama/mistral-small3.1:24b",
        "messages": [
            {"role": "user", "content": "Whats the weather in Tokyo?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
    }' \
    -w "%{http_code}")

FC_HTTP_CODE=${FC_RESPONSE: -3}
FC_RESPONSE_BODY=${FC_RESPONSE:0:${#FC_RESPONSE}-3}

if [[ "$FC_HTTP_CODE" == "200" ]]; then
    echo -e "${GREEN}Function calling endpoint working!${NC}"
    # Check if there's a function call in the response
    if [[ "$FC_RESPONSE_BODY" == *"function_call"* || "$FC_RESPONSE_BODY" == *"tool_calls"* ]]; then
        echo -e "${GREEN}Success!${NC} Function calling is working correctly."
    else
        echo -e "${YELLOW}Warning!${NC} Endpoint worked but no function call was generated."
        echo "This might be normal depending on the model behavior."
    fi
else
    echo -e "${RED}Failed!${NC} Function calling test returned HTTP code $FC_HTTP_CODE"
    echo "Response body:"
    echo "$FC_RESPONSE_BODY" | head -20
fi

echo -e "\n${YELLOW}Diagnosis complete.${NC}"
echo -e "Use this information to update the tests in ${GREEN}tests/conftest.py${NC} with working models."
