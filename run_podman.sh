#!/bin/bash

# Script to build and run the model-calling service with Podman

# Set colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building model-calling service with Podman...${NC}"
podman build -t localhost/model-calling:latest -f Containerfile .

echo -e "${GREEN}Running model-calling service...${NC}"
echo -e "${YELLOW}To access the service, go to: http://localhost:8000${NC}"

# Run in foreground mode - use CTRL+C to stop
podman run --rm -it \
  -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.containers.internal:11434 \
  --name model-calling \
  localhost/model-calling:latest

# Alternatively, you can use podman-compose:
# echo -e "${GREEN}Starting services with podman-compose...${NC}"
# podman-compose -f podman-compose.yml up
