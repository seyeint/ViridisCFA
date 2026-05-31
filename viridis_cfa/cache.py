"""
Cache infrastructure for ViridisCFA ingredient-based pipeline caching.

Each ticker gets a manifest at data/cache/{TICKER}.json that tracks:
- The "ingredients" that went into the analysis (filing accession_no,
  quant engine version, report UI versions, transcript hash, insider hash)
- Paths to all generated artifacts
- A history of runs with cost and cache-hit information

The pipeline compares current ingredients against the manifest to determine
which LLM steps need re-running vs. which can be served from cache.
"""

import os
import json
import hashlib
from datetime import datetime, timezone

CACHE_DIR = os.path.join("data", "cache")

# Stable marker stored as the transcript "hash" when a scrape attempt completed but
# found no transcript. Distinguishes "checked, none available" from "never checked",
# so a no-transcript ticker reaches a steady cache state instead of re-running the
# paid final synthesis (and re-scraping) on every single run.
NO_TRANSCRIPT_SENTINEL = "__no_transcript__"


def compute_hash(content: str) -> str:
    """SHA-256 hash of a content string. Returns hex digest."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_manifest(ticker: str) -> dict | None:
    """Load the cached manifest for a ticker. Returns None if not found or corrupt."""
    path = os.path.join(CACHE_DIR, f"{ticker.upper()}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[CACHE] Failed to read manifest for {ticker}: {e}")
        return None


def save_manifest(ticker: str, manifest: dict) -> None:
    """Write a manifest to disk, creating the cache directory if needed."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{ticker.upper()}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)


def check_artifacts_exist(manifest: dict) -> bool:
    """Verify that all artifact files referenced by the manifest exist on disk.
    
    Returns False if any required artifact is missing, which means the manifest
    should be treated as invalid and a fresh run is needed.
    """
    artifacts = manifest.get("artifacts", {})
    for key, path in artifacts.items():
        if path and not os.path.exists(path):
            print(f"[CACHE] Artifact missing from disk: {path}")
            return False
    return True


def list_all_manifests() -> list[dict]:
    """List all known analyses — from manifests and from pre-cache report files.
    
    Returns a list of dicts with keys:
        ticker, filing_form, filing_date, analyzed_at, total_cost, last_cost, has_manifest
    Sorted by analyzed_at descending.
    """
    results = []
    seen_tickers = set()

    # 1. Scan cache manifests
    if os.path.exists(CACHE_DIR):
        for filename in sorted(os.listdir(CACHE_DIR)):
            if not filename.endswith(".json"):
                continue
            try:
                with open(os.path.join(CACHE_DIR, filename), "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                ticker = manifest.get("ticker", filename.replace(".json", ""))
                seen_tickers.add(ticker.upper())

                runs = manifest.get("runs", [])
                latest_run = runs[-1] if runs else {}

                results.append({
                    "ticker": ticker,
                    "filing_form": manifest.get("ingredients", {}).get("filing_form", "N/A"),
                    "filing_date": manifest.get("ingredients", {}).get("filing_date", "N/A"),
                    "analyzed_at": latest_run.get("timestamp", "N/A"),
                    "total_cost": sum(r.get("cost", 0) for r in runs),
                    "last_cost": latest_run.get("cost", 0),
                    "has_manifest": True,
                })
            except Exception:
                continue

    # 2. Fallback: scan data/intermediate/ for pre-cache reports (backward compat)
    intermediate_dir = os.path.join("data", "intermediate")
    if os.path.exists(intermediate_dir):
        for filename in sorted(os.listdir(intermediate_dir)):
            if not filename.endswith("_final_report.md"):
                continue
            ticker = filename.replace("_final_report.md", "").upper()
            if ticker in seen_tickers:
                continue  # Already found via manifest

            filepath = os.path.join(intermediate_dir, filename)
            mod_time = os.path.getmtime(filepath)
            mod_date = datetime.fromtimestamp(mod_time, tz=timezone.utc).isoformat()

            results.append({
                "ticker": ticker,
                "filing_form": "N/A",
                "filing_date": "N/A",
                "analyzed_at": mod_date,
                "total_cost": 0,
                "last_cost": 0,
                "has_manifest": False,
            })

    # Sort by analyzed_at descending
    results.sort(key=lambda x: x.get("analyzed_at", ""), reverse=True)
    return results


def now_iso() -> str:
    """Current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()
