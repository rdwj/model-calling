#!/bin/bash

# Make all shell scripts executable
chmod +x start_service.sh
chmod +x test_ollama.sh
chmod +x run_podman.sh
chmod +x run_podman_compose.sh
chmod +x run_tests.sh
chmod +x diagnose_models.sh
chmod +x auto_config_tests.sh
chmod +x fix_test_imports.sh

echo "All shell scripts are now executable!"
