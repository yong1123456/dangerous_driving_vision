#!/usr/bin/env python3
"""Download a cataloged evaluation video without weakening its split rules."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from audit_video_catalog import load_catalog, probe_video, resolve_local_path, sha256_file


DOWNLOAD_GROUPS = ("downloaded_sources", "open_video_candidates")


def find_download_entry(catalog: dict[str, Any], entry_id: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for group in DOWNLOAD_GROUPS:
        for entry in catalog.get(group, []):
            if isinstance(entry, dict) and str(entry.get("id")) == entry_id:
                matches.append({"catalog_group": group, **entry})
    if not matches:
        raise KeyError(f"catalog video ID not found or not downloadable: {entry_id}")
    if len(matches) > 1:
        raise ValueError(f"catalog video ID is not unique: {entry_id}")
    return matches[0]


def _destination(entry: dict[str, Any], root: Path) -> Path:
    destination = resolve_local_path(entry, root)
    if destination is None:
        raise ValueError("catalog entry has no local_path")
    root_resolved = root.resolve()
    destination_resolved = destination.resolve()
    try:
        destination_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("download destination must stay inside the catalog root") from exc
    return destination_resolved


def _validate_partial(
    entry: dict[str, Any],
    partial: Path,
    video_probe: Callable[[Path], dict[str, Any] | None],
) -> tuple[str, dict[str, Any]]:
    digest = sha256_file(partial)
    expected = str(entry.get("sha256", "")).lower()
    if expected and digest != expected:
        raise RuntimeError(f"SHA-256 mismatch: expected {expected}, got {digest}")
    metadata = video_probe(partial)
    if metadata is None:
        raise RuntimeError("downloaded file is not a decodable video")
    return digest, metadata


def _download_with_curl(
    url: str,
    partial: Path,
    *,
    timeout: float,
    resume: bool,
    runner: Callable[..., Any],
) -> None:
    command = [
        "curl.exe",
        "-L",
        "-4",
        "--http1.1",
        "--ssl-no-revoke",
        "--fail",
        "--retry",
        "3",
        "--retry-all-errors",
        "--connect-timeout",
        str(timeout),
        "--speed-time",
        "120",
        "--speed-limit",
        "1024",
    ]
    if resume and partial.is_file():
        command.extend(["--continue-at", "-"])
    command.extend(["--output", str(partial), url])
    try:
        completed = runner(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"curl transport could not start: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "unknown curl error").strip().splitlines()
        raise RuntimeError(f"curl download failed: {detail[-1] if detail else 'unknown error'}")


def download_entry(
    entry: dict[str, Any],
    *,
    root: Path,
    timeout: float = 30.0,
    resume: bool = True,
    force: bool = False,
    transport: str = "auto",
    opener: Callable[..., Any] = urlopen,
    curl_runner: Callable[..., Any] = subprocess.run,
    video_probe: Callable[[Path], dict[str, Any] | None] = probe_video,
) -> dict[str, Any]:
    if transport not in {"auto", "python", "curl"}:
        raise ValueError("transport must be auto, python, or curl")
    url = str(entry.get("direct_url", ""))
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("catalog entry must have an HTTPS direct_url")
    destination = _destination(entry, root)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and not force:
        digest = sha256_file(destination)
        expected = str(entry.get("sha256", "")).lower()
        if expected and digest != expected:
            raise RuntimeError("existing destination does not match the catalog SHA-256")
        metadata = video_probe(destination)
        if metadata is None:
            raise RuntimeError("existing destination is not a decodable video; use --force to replace it")
        return {
            "status": "already_exists",
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": digest,
            "video": metadata,
        }

    partial = destination.with_suffix(destination.suffix + ".part")
    existing_bytes = partial.stat().st_size if resume and partial.is_file() and not force else 0
    used_transport = "curl" if transport == "curl" else "python"
    python_error: str | None = None
    if transport in {"auto", "python"}:
        headers = {"User-Agent": "dangerous-driving-vision-catalog-downloader/1.0"}
        if existing_bytes:
            headers["Range"] = f"bytes={existing_bytes}-"
        request = Request(url, headers=headers)
        try:
            response_context = opener(request, timeout=timeout)
            with response_context as response:
                status = getattr(response, "status", None) or response.getcode()
                if status not in (200, 206):
                    raise RuntimeError(f"unexpected HTTP status: {status}")
                append = bool(existing_bytes and status == 206)
                mode = "ab" if append else "wb"
                with partial.open(mode) as stream:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
        except HTTPError as exc:
            raise RuntimeError(f"HTTP download failed: {exc.code}") from exc
        except OSError as exc:
            if transport == "python":
                raise RuntimeError(f"Python HTTPS download failed: {exc}") from exc
            python_error = str(exc)
            used_transport = "curl"

    if used_transport == "curl":
        _download_with_curl(
            url,
            partial,
            timeout=timeout,
            resume=resume and not force,
            runner=curl_runner,
        )

    digest, metadata = _validate_partial(entry, partial, video_probe)
    partial.replace(destination)
    manifest = {
        "schema_version": 1,
        "id": entry.get("id"),
        "source_url": url,
        "page_url": entry.get("page_url"),
        "rights_status": entry.get("rights_status"),
        "use": entry.get("use"),
        "split": entry.get("split"),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "transport": used_transport,
        "bytes": destination.stat().st_size,
        "sha256": digest,
        "video": metadata,
    }
    sidecar = destination.with_suffix(destination.suffix + ".download.json")
    sidecar.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {"status": "downloaded", "path": str(destination), **manifest}
    if python_error:
        result["python_https_error"] = python_error
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, help="Catalog video ID")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "china_video_catalog.json",
    )
    parser.add_argument("--root", type=Path, help="Base directory for relative catalog paths")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--transport", choices=("auto", "python", "curl"), default="auto")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    entry = find_download_entry(catalog, args.id)
    root = args.root or args.catalog.resolve().parent.parent
    print(
        f"id={entry.get('id')} split={entry.get('split')} use={entry.get('use')} "
        f"rights={entry.get('rights_status')}"
    )
    print("notice=Downloaded media keeps its cataloged split and is not automatically training data.")
    try:
        result = download_entry(
            entry,
            root=root,
            timeout=args.timeout,
            resume=not args.no_resume,
            force=args.force,
            transport=args.transport,
        )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        print(f"error={exc}", file=sys.stderr)
        return 1
    print(f"status={result['status']}")
    if result.get("python_https_error"):
        print(f"python_https_error={result['python_https_error']}")
    print(f"transport={result.get('transport', 'existing-file')}")
    print(f"path={result['path']}")
    print(f"bytes={result['bytes']}")
    print(f"sha256={result['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
