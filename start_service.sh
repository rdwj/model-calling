#!/bin/bash

# Start the model-calling service
echo "Starting model-calling service on port 8000..."
python -m model_calling.server
