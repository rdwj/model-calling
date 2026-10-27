"""
Example of using reasoning capabilities with model-calling service.
"""
import asyncio
from model_calling.client import ModelCallingClient


async def math_reasoning_example():
    """Example of using a model for mathematical reasoning"""
    client = ModelCallingClient()
    
    # Math problem that requires step-by-step reasoning
    math_problem = (
        "If a train travels at 60 mph for 3 hours and then at 80 mph for 2 hours, "
        "what is the average speed for the entire journey?"
    )
    
    try:
        # Call the reasoning-capable model
        print("Math Reasoning Problem:")
        print(math_problem)
        print("\nSolving with step-by-step reasoning...")
        
        response = await client.chat_completion(
            model="ollama/phi4-mini-reasoning:3.8b",  # Using phi4-mini-reasoning model
            messages=[
                {
                    "role": "user", 
                    "content": f"Solve this step by step: {math_problem}"
                }
            ]
        )
        
        # Extract and print the solution
        solution = response["choices"][0]["message"]["content"]
        print("\nSolution:")
        print(solution)
        
        return solution
    
    except Exception as e:
        print(f"Error solving math problem: {e}")
        # Fallback to another model if the first one fails
        print("\nFalling back to alternate model...")
        
        response = await client.chat_completion(
            model="ollama/mistral:latest",  # Fallback to Mistral
            messages=[
                {
                    "role": "user", 
                    "content": f"Solve this step by step: {math_problem}"
                }
            ]
        )
        
        solution = response["choices"][0]["message"]["content"]
        print("\nSolution (fallback model):")
        print(solution)
        
        return solution


async def logical_reasoning_example():
    """Example of using a model for logical reasoning"""
    client = ModelCallingClient()
    
    # Logical reasoning problem
    logic_problem = (
        "Three friends - Alex, Bailey, and Casey - each have a different pet: a dog, a cat, and a bird. "
        "Alex does not have the bird. Bailey does not have the dog. Casey does not have the cat. "
        "Who has which pet?"
    )
    
    try:
        # Call the reasoning-capable model
        print("Logical Reasoning Problem:")
        print(logic_problem)
        print("\nSolving with logical reasoning...")
        
        response = await client.chat_completion(
            model="ollama/phi4-mini-reasoning:3.8b",  # Using phi4-mini-reasoning model
            messages=[
                {
                    "role": "user", 
                    "content": f"Think through this logical puzzle step by step: {logic_problem}"
                }
            ]
        )
        
        # Extract and print the solution
        solution = response["choices"][0]["message"]["content"]
        print("\nSolution:")
        print(solution)
        
        return solution
    
    except Exception as e:
        print(f"Error solving logical problem: {e}")
        return None


async def planning_example():
    """Example of using a model for planning tasks"""
    client = ModelCallingClient()
    
    # Planning task
    planning_task = (
        "Create a step-by-step plan for organizing a small conference for 50 people. "
        "Include timeline, budget considerations, and key tasks."
    )
    
    try:
        # Call the reasoning-capable model
        print("Planning Task:")
        print(planning_task)
        print("\nGenerating plan...")
        
        response = await client.chat_completion(
            model="ollama/phi4-mini-reasoning:3.8b",  # Using phi4-mini-reasoning model
            messages=[
                {
                    "role": "user", 
                    "content": planning_task
                }
            ]
        )
        
        # Extract and print the plan
        plan = response["choices"][0]["message"]["content"]
        print("\nGenerated Plan:")
        print(plan)
        
        return plan
    
    except Exception as e:
        print(f"Error generating plan: {e}")
        return None


async def main():
    """Run all examples"""
    print("=" * 50)
    print("Reasoning Capabilities Examples")
    print("=" * 50)
    
    print("\n1. Mathematical Reasoning Example")
    print("-" * 30)
    await math_reasoning_example()
    
    print("\n" + "=" * 50 + "\n")
    
    print("2. Logical Reasoning Example")
    print("-" * 30)
    await logical_reasoning_example()
    
    print("\n" + "=" * 50 + "\n")
    
    print("3. Planning Example")
    print("-" * 30)
    await planning_example()


if __name__ == "__main__":
    asyncio.run(main())
