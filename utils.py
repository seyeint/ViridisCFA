import tiktoken
from typing import Dict

# Token counting function
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in the text using the encoding for the specified model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"WARNING: Unknown model '{model}'. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))

# Model pricing constants (per 1M tokens)
MODEL_PRICING = {
    "gpt-5.4": {
        "input": 2.50,   # $2.50 per 1M tokens
        "output": 15.00  # $15.00 per 1M tokens
    },
    "gpt-4o": {
        "input": 2.50,   # $2.50 per 1M tokens
        "output": 10.00  # $10.00 per 1M tokens
    },
    "gpt-4o-mini": {
        "input": 0.15,   # $0.15 per 1M tokens
        "output": 0.60   # $0.60 per 1M tokens
    },
    "o3-mini": {
        "input": 1.10,   # $1.10 per 1M tokens
        "output": 4.40   # $4.40 per 1M tokens
    },
    "o4-mini": {
        "input": 1.10,   # $1.10 per 1M tokens
        "output": 4.40   # $4.40 per 1M tokens
    }
}

def estimate_cost(tokens: int, model: str, output_ratio: float = 0.2) -> Dict[str, float]:
    """Estimate cost based on input tokens and model. Output ratio is probably overestimated."""
    pricing = MODEL_PRICING.get(model, {"input": 10.0, "output": 30.0})
    
    estimated_output_tokens = round(tokens * output_ratio)
    input_cost = (tokens / 1000000) * pricing["input"]
    output_cost = (estimated_output_tokens / 1000000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

def calculate_actual_cost(prompt_tokens: int, completion_tokens: int, model: str) -> Dict[str, float]:
    """Calculate actual cost from API response token usage."""
    pricing = MODEL_PRICING.get(model, {"input": 10.0, "output": 30.0})
    
    input_cost = (prompt_tokens / 1000000) * pricing["input"]
    output_cost = (completion_tokens / 1000000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    } 