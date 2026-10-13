#!/usr/bin/env python3
"""Search GitHub for candidate road-vision datasets without downloading them."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_QUERIES = [
    "Chinese dashcam dataset",
    "China traffic light dataset",
    "Chinese lane detection dataset",
    "traffic violation video dataset",
]


def github_search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
        {"q": query, "per_page": min(max(limit, 1), 30)}
    )
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "dangerous-driving-vision"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    rows: list[dict[str, Any]] = []
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        license_info = item.get("license") or {}
        rows.append(
            {
                "full_name": item.get("full_name"),
                "html_url": item.get("html_url"),
                "description": item.get("description"),
                "stars": item.get("stargazers_count"),
                "license": license_info.get("spdx_id"),
                "updated_at": item.get("updated_at"),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", dest="queries", help="GitHub query; repeatable")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    queries = args.queries or DEFAULT_QUERIES
    results: list[dict[str, Any]] = []
    for query in queries:
        try:
            repositories = github_search(query, args.limit)
            results.append({"query": query, "status": "ok", "repositories": repositories})
            print(f"query={query} results={len(repositories)}")
        except Exception as error:  # network/rate-limit errors are per-query
            results.append({"query": query, "status": "error", "error": str(error), "repositories": []})
            print(f"query={query} status=error error={error}")
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "queries": results,
        "notice": "Search results are candidates only. Inspect the official dataset terms before download or training.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
