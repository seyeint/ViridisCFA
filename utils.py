import tiktoken
from typing import Dict, Tuple

# Token counting function
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in the text using the encoding for the specified model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"WARNING: Unknown model '{model}'. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))

# Pricing for the active pipeline model, per 1M tokens.
# Keep this narrow; add other models only when the pipeline actually uses them.
MODEL_PRICING = {
    "gpt-5.4": {
        "standard": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
        "default": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
        "auto": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
        "flex": {"input": 1.25, "cached_input": 0.125, "output": 7.50},
        "batch": {"input": 1.25, "cached_input": 0.125, "output": 7.50},
        "priority": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
        "long_context_threshold": 272_000,
        "long_context": {
            "standard": {"input": 5.00, "cached_input": 0.50, "output": 22.50},
            "default": {"input": 5.00, "cached_input": 0.50, "output": 22.50},
            "auto": {"input": 5.00, "cached_input": 0.50, "output": 22.50},
            "flex": {"input": 2.50, "cached_input": 0.25, "output": 11.25},
            "batch": {"input": 2.50, "cached_input": 0.25, "output": 11.25},
        },
    }
}

def _normalize_model_name(model: str) -> str:
    """Map dated model snapshots back to the base pricing key."""
    if model in MODEL_PRICING:
        return model
    for base_model in sorted(MODEL_PRICING.keys(), key=len, reverse=True):
        if model.startswith(base_model):
            return base_model
    return model


def _pricing_for(model: str, service_tier: str = "standard", input_tokens: int = 0) -> Tuple[Dict[str, float], str, str]:
    """Return pricing table, normalized model, and context band."""
    normalized_model = _normalize_model_name(model)
    model_pricing = MODEL_PRICING.get(normalized_model)
    if not model_pricing:
        return {"input": 10.0, "cached_input": 10.0, "output": 30.0}, normalized_model, "unknown"

    tier = (service_tier or "standard").lower()
    context_band = "short"
    threshold = model_pricing.get("long_context_threshold")
    if threshold and input_tokens > threshold and "long_context" in model_pricing:
        long_context = model_pricing["long_context"]
        if tier in long_context:
            context_band = "long"
            return long_context[tier], normalized_model, context_band

    if tier not in model_pricing:
        tier = "standard"
    return model_pricing[tier], normalized_model, context_band


def estimate_cost(tokens: int, model: str, output_ratio: float = 0.2, service_tier: str = "standard") -> Dict[str, float]:
    """Estimate cost based on input tokens and model. Output ratio is probably overestimated."""
    pricing, normalized_model, context_band = _pricing_for(model, service_tier, tokens)
    
    estimated_output_tokens = round(tokens * output_ratio)
    input_cost = (tokens / 1000000) * pricing["input"]
    output_cost = (estimated_output_tokens / 1000000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "service_tier": service_tier,
        "model": normalized_model,
        "context_band": context_band,
    }

def calculate_actual_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    service_tier: str = "standard",
    cached_input_tokens: int = 0,
) -> Dict[str, float]:
    """Calculate actual cost from API response token usage."""
    pricing, normalized_model, context_band = _pricing_for(model, service_tier, prompt_tokens)
    
    cached_tokens = min(max(cached_input_tokens or 0, 0), prompt_tokens)
    uncached_tokens = max(prompt_tokens - cached_tokens, 0)

    uncached_input_cost = (uncached_tokens / 1000000) * pricing["input"]
    cached_input_cost = (cached_tokens / 1000000) * pricing["cached_input"]
    input_cost = uncached_input_cost + cached_input_cost
    output_cost = (completion_tokens / 1000000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "uncached_input_cost": uncached_input_cost,
        "cached_input_cost": cached_input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "service_tier": service_tier,
        "model": normalized_model,
        "context_band": context_band,
    }
