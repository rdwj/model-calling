#!/bin/bash

# Script to run the model-calling service with podman-compose

# Set colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting model-calling services with podman-compose...${NC}"
echo -e "${YELLOW}To access the service, go to: http://localhost:8000${NC}"

# Check if podman-compose is installed
if ! command -v podman-compose &> /dev/null; then
    echo -e "${YELLOW}podman-compose not found. Installing it first...${NC}"
    pip install podman-compose==1.6.0
fi

# Run the services
podman-compose -f podman-compose.yml up

# To stop, use CTRL+C or run in another terminal:
# podman-compose -f podman-compose.yml down
