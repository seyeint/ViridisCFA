import json
import os
import re
from typing import Any, Dict, List, Optional


DECISION_BRIEF_VERSION = "1.1"

STANCE_VALUES = {"Bullish", "Neutral", "Bearish", "Mixed"}
CONVICTION_VALUES = {"High", "Medium", "Low", "N/A"}
SERIALIZED_FIELD_PATTERN = re.compile(
    r"""["']?(schema_version|ticker|stance|conviction|thesis|upside_drivers|risk_drivers|key_catalysts|data_quality_flags)["']?\s*:""",
    re.IGNORECASE,
)
CORRUPTION_MARKERS = (
    "```",
    "Oops,",
    "Here is the corrected JSON",
    "to=final",
)

DECISION_BRIEF_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "ticker",
        "stance",
        "conviction",
        "thesis",
        "upside_drivers",
        "risk_drivers",
        "key_catalysts",
        "data_quality_flags",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "ticker": {"type": "string"},
        "stance": {"type": "string", "enum": sorted(STANCE_VALUES)},
        "conviction": {"type": "string", "enum": sorted(CONVICTION_VALUES)},
        "thesis": {"type": "string"},
        "upside_drivers": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 5,
        },
        "risk_drivers": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 5,
        },
        "key_catalysts": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 5,
        },
        "data_quality_flags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 5,
        },
    },
}


def build_decision_brief_prompt(ticker: str, final_report_markdown: str) -> str:
    return f"""You are producing a structured UI contract for an investment research report.

Read the final report below and return ONLY a JSON object matching the provided schema.

Rules:
- Use only information explicitly present in the report.
- Do not add valuation, market prices, estimates, or external facts.
- Keep each list item concise but specific.
- Each list item must be plain natural language. Never pack multiple JSON/list items into one string.
- Do not include serialized JSON, braces, brackets, escaped quotes, or schema field names inside string values.
- If a bucket truly has no support in the report, return an empty list for that bucket.
- `stance` must be Bullish, Neutral, Bearish, or Mixed.
- `conviction` must be High, Medium, Low, or N/A.
- `thesis` should be a single sentence or compact paragraph capturing the investment case.
- `data_quality_flags` should include timing mismatches, missing data, derived/imputed quant inputs, accounting/control issues, or provenance caveats when present.

Ticker: {ticker}

Final report markdown:

{final_report_markdown}
"""


def _normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    if not isinstance(value, str):
        return default
    cleaned = value.strip().title()
    for allowed_value in allowed:
        if cleaned.lower() == allowed_value.lower():
            return allowed_value
    return default


def _normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _trim_text_item(text: str, max_chars: int = 320) -> str:
    if len(text) <= max_chars:
        return text
    for marker in (". ", "; ", ", "):
        cut = text.rfind(marker, 0, max_chars)
        if cut >= 120:
            return text[: cut + 1].strip()
    return text[:max_chars].rstrip() + "..."


def _clean_list_item(text: str) -> str:
    for marker in CORRUPTION_MARKERS:
        marker_index = text.find(marker)
        if marker_index >= 0:
            text = text[:marker_index]

    field_match = SERIALIZED_FIELD_PATTERN.search(text)
    if field_match:
        text = text[:field_match.start()]

    text = text.strip()
    text = re.sub(r"""^[\s,\]\}\{"']+""", "", text)
    text = re.sub(r"""[\s,\[\]\{\}"']+$""", "", text)
    text = re.sub(r"\s+", " ", text)
    return _trim_text_item(text)


def _split_packed_list_item(text: str) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []

    items = []
    for fragment in re.split(r"""["']\s*,\s*["']""", text):
        stop_after_fragment = (
            bool(SERIALIZED_FIELD_PATTERN.search(fragment)) or
            any(marker in fragment for marker in CORRUPTION_MARKERS)
        )
        cleaned = _clean_list_item(fragment)
        if cleaned:
            items.append(cleaned)
        if stop_after_fragment:
            break
    return items


def _normalize_list(value: Any, limit: int = 5) -> List[str]:
    if not isinstance(value, list):
        return []
    items = []
    seen = set()
    for item in value:
        for text in _split_packed_list_item(item):
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(text)
            if len(items) >= limit:
                break
        if len(items) >= limit:
            break
    return items


def normalize_decision_brief(raw: Any, ticker: str) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    return {
        "schema_version": DECISION_BRIEF_VERSION,
        "ticker": _normalize_text(raw.get("ticker"), ticker).upper(),
        "stance": _normalize_choice(raw.get("stance"), STANCE_VALUES, "Neutral"),
        "conviction": _normalize_choice(raw.get("conviction"), CONVICTION_VALUES, "N/A"),
        "thesis": _normalize_text(
            raw.get("thesis"),
            "Decision brief unavailable. Read the full report for the synthesized investment view.",
        ),
        "upside_drivers": _normalize_list(raw.get("upside_drivers")),
        "risk_drivers": _normalize_list(raw.get("risk_drivers")),
        "key_catalysts": _normalize_list(raw.get("key_catalysts")),
        "data_quality_flags": _normalize_list(raw.get("data_quality_flags")),
    }


def parse_decision_brief_json(text: str, ticker: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return normalize_decision_brief(json.loads(cleaned), ticker)


FENCED_CODE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_and_strip_brief_json(memo_text: str):
    """Pull the trailing fenced JSON decision brief out of a final memo that emitted
    both the prose and the brief in one call. Returns ``(memo_without_block, json_str)``
    — picking the last fenced block whose content looks like a JSON object — or
    ``(memo_text, None)`` when there is no usable block, so the caller can fall back to a
    separate structuring call."""
    if not memo_text:
        return memo_text, None
    for match in reversed(list(FENCED_CODE_RE.finditer(memo_text))):
        content = match.group(1).strip()
        if content.startswith("{") and content.endswith("}"):
            stripped = (memo_text[:match.start()] + memo_text[match.end():]).rstrip()
            return stripped, content
    return memo_text, None


def load_decision_brief(path: Optional[str], ticker: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return normalize_decision_brief(json.load(f), ticker)
    except (OSError, json.JSONDecodeError):
        return None


def write_decision_brief(path: str, brief: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ticker = brief.get("ticker", "") if isinstance(brief, dict) else ""
    clean_brief = normalize_decision_brief(brief, ticker)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_brief, f, indent=2, ensure_ascii=False)
