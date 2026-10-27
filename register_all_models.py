#!/usr/bin/env python3
"""
Register and test all local Ollama models with the new comprehensive capability testing system.
This script will:
1. List all available Ollama models
2. Register and test each model's capabilities
3. Display a summary of all model capabilities
"""
import asyncio
import json
import logging
import sys
import time
from typing import Dict, List, Any, Optional
import httpx

from model_calling.registry import registry
from model_calling.capability_cache import capability_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("register_models")

OLLAMA_API_BASE = "http://localhost:11434"

async def get_all_ollama_models() -> List[str]:
    """Get all available Ollama models"""
    logger.info("Fetching all available Ollama models...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_API_BASE}/api/tags", timeout=30.0)
            response.raise_for_status()
            
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            
            logger.info(f"Found {len(models)} Ollama models")
            return models
    except Exception as e:
        logger.error(f"Failed to fetch Ollama models: {e}")
        return []

async def register_and_test_model(model_name: str, force_retest: bool = False, timeout: int = 120) -> Dict[str, Any]:
    """
    Register and test a model's capabilities
    
    Args:
        model_name: The model name (without ollama/ prefix)
        force_retest: If True, ignore cache and force a fresh test
        timeout: Maximum time in seconds to wait for testing to complete
    
    Returns:
        Dict with test results
    """
    start_time = time.time()
    model_id = f"ollama/{model_name}"
    logger.info(f"Registering and testing model: {model_id}")
    
    try:
        # Create a task with timeout
        verification_task = asyncio.create_task(registry.verify_capabilities(model_id, force_retest=force_retest))
        capabilities = await asyncio.wait_for(verification_task, timeout=timeout)
        elapsed_time = time.time() - start_time
    except asyncio.TimeoutError:
        logger.warning(f"Testing {model_id} timed out after {timeout} seconds")
        # Create a basic capability object with defaults
        capabilities = registry.ModelCapabilities()
        capabilities.supports_streaming = True
        capabilities.supports_system_messages = True
        capabilities.verified = False
        elapsed_time = timeout
    
    # Convert to dict for reporting
    capabilities_dict = capabilities.to_dict()
    
    # Create result summary
    result = {
        "model": model_id,
        "capabilities": capabilities_dict,
        "test_duration_seconds": elapsed_time,
    }
    
    # Display capability summary
    logger.info(f"\nCapabilities for {model_id}:")
    logger.info(f"  Function calling: {'✅' if capabilities_dict['supports_function_call'] else '❌'}")
    logger.info(f"  JSON mode: {'✅' if capabilities_dict['supports_json_mode'] else '❌'}")
    logger.info(f"  Reasoning: {'✅' if capabilities_dict['supports_reasoning'] else '❌'}")
    logger.info(f"  Vision: {'✅' if capabilities_dict['supports_vision'] else '❌'}")
    logger.info(f"  Thinking style: {capabilities_dict['thinking_style']}")
    logger.info(f"  Verified: {'✅' if capabilities_dict['verified'] else '❌'}")
    logger.info(f"  Test duration: {elapsed_time:.1f} seconds")
    
    return result

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Register and test all Ollama models")
    parser.add_argument("--force", "-f", action="store_true", help="Force retesting, ignore cache")
    parser.add_argument("--models", "-m", nargs="+", help="Specific models to test (without ollama/ prefix)")
    parser.add_argument("--timeout", "-t", type=int, default=120, help="Timeout in seconds for testing each model")
    parser.add_argument("--parallel", "-p", type=int, default=1, help="Number of models to test in parallel")
    args = parser.parse_args()
    
    # Get models to test
    if args.models:
        models_to_test = args.models
    else:
        models_to_test = await get_all_ollama_models()
    
    if not models_to_test:
        logger.error("No models found to test. Is Ollama running?")
        return
    
    logger.info(f"Registering and testing {len(models_to_test)} models...")
    
    # Test models (in parallel if requested)
    results = []
    
    if args.parallel > 1:
        # Process models in batches for parallel testing
        for i in range(0, len(models_to_test), args.parallel):
            batch = models_to_test[i:i+args.parallel]
            logger.info(f"Testing batch of {len(batch)} models in parallel")
            
            tasks = [register_and_test_model(model, args.force, args.timeout) for model in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for model, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to test {model}: {result}")
                else:
                    results.append(result)
    else:
        # Test models sequentially
        for model_name in models_to_test:
            try:
                result = await register_and_test_model(model_name, args.force, args.timeout)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to test {model_name}: {e}")
    
    # Display summary table
    print("\n=== MODEL CAPABILITIES SUMMARY ===")
    print(f"{'Model':<30} | {'Func Call':<10} | {'JSON':<6} | {'Reasoning':<10} | {'Vision':<6} | {'Verified':<8}")
    print("-" * 80)
    
    for result in results:
        model = result["model"].split("/")[-1]  # Just show model name without prefix
        caps = result["capabilities"]
        print(f"{model:<30} | {'✅' if caps['supports_function_call'] else '❌':<10} | "
              f"{'✅' if caps['supports_json_mode'] else '❌':<6} | "
              f"{'✅' if caps['supports_reasoning'] else '❌':<10} | "
              f"{'✅' if caps['supports_vision'] else '❌':<6} | "
              f"{'✅' if caps['verified'] else '❌':<8}")
    
    # Save results to a JSON file for reference
    with open("model_capabilities.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRegistered and tested {len(results)} models.")
    print("Results saved to model_capabilities.json")

if __name__ == "__main__":
    asyncio.run(main())
