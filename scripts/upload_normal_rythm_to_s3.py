from __future__ import annotations

import argparse
import mimetypes
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = "phrasebook-audio-files"
DEFAULT_REGION = "us-east-1"


def _local_normal_rythm_dir(language: str) -> Path:
    lang = language.strip().title()
    return (
        REPO_ROOT
        / "private_assets"
        / "Languages"
        / f"{lang}Phrasebook"
        / "Results_Audios_normal_pace"
        / "normal_rythm"
    )


def _guess_content_type(path: Path) -> str:
    ctype, _ = mimetypes.guess_type(str(path))
    return ctype or "application/octet-stream"


def _list_existing_keys(s3, bucket: str, prefix: str) -> set[str]:
    existing: set[str] = set()
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []) or []:
            k = obj.get("Key")
            if k:
                existing.add(str(k))
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
            continue
        break
    return existing


def _run_aws_cli_sync(
    *,
    local_dir: Path,
    bucket: str,
    prefix: str,
    region: str,
    profile: str,
    dry_run: bool,
    size_only: bool,
) -> int:
    aws = shutil.which("aws")
    if not aws:
        print("Error: AWS CLI not found on PATH (aws).", file=sys.stderr)
        return 2

    dest = f"s3://{bucket}/{prefix}"
    cmd = [aws, "s3", "sync", str(local_dir), dest, "--only-show-errors", "--no-progress"]
    if dry_run:
        cmd.append("--dryrun")
    if size_only:
        cmd.append("--size-only")
    if profile:
        cmd += ["--profile", profile]
    if region:
        cmd += ["--region", region]

    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd)
    return int(res.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Upload <Lang> Results_Audios_normal_pace/normal_rythm to S3, skipping files already present.\n"
            "Requires AWS credentials configured for boto3."
        )
    )
    ap.add_argument("--language", required=True, help="Language name, e.g. Yemba, Ewondo.")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"S3 bucket name (default: {DEFAULT_BUCKET}).")
    ap.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", DEFAULT_REGION)),
        help=f"AWS region (default: {DEFAULT_REGION}).",
    )
    ap.add_argument(
        "--prefix",
        default="",
        help="S3 key prefix (default: 'audio_<language.lower()>/').",
    )
    ap.add_argument("--workers", type=int, default=8, help="Parallel uploads (default: 8).")
    ap.add_argument("--dry-run", action="store_true", help="Show what would upload but do not upload.")
    ap.add_argument("--profile", default="", help="Optional AWS profile name.")
    ap.add_argument(
        "--sync",
        action="store_true",
        help="Use AWS CLI 'aws s3 sync' (recommended for large uploads).",
    )
    ap.add_argument(
        "--size-only",
        action="store_true",
        help="(sync mode) Skip uploads when size matches (cheapest/fastest).",
    )
    args = ap.parse_args()

    language = args.language.strip()
    if not language:
        print("Error: --language is required", file=sys.stderr)
        return 2

    local_dir = _local_normal_rythm_dir(language)
    if not local_dir.exists():
        print(f"Error: local directory not found: {local_dir}", file=sys.stderr)
        return 2

    prefix = args.prefix.strip()
    if not prefix:
        prefix = f"audio_{language.lower()}/"
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    # Recommended path for large uploads: AWS CLI sync.
    if args.sync:
        return _run_aws_cli_sync(
            local_dir=local_dir,
            bucket=args.bucket,
            prefix=prefix,
            region=args.region,
            profile=args.profile,
            dry_run=args.dry_run,
            size_only=bool(args.size_only),
        )

    try:
        import boto3  # type: ignore
    except Exception:
        print("Error: boto3 is not installed. Install with: pip install boto3", file=sys.stderr)
        return 2

    session_kwargs = {}
    if args.profile:
        session_kwargs["profile_name"] = args.profile
    session = boto3.Session(**session_kwargs)
    s3 = session.client("s3", region_name=args.region)

    files = sorted([p for p in local_dir.iterdir() if p.is_file()], key=lambda p: p.name.lower())
    if not files:
        print(f"No files found in {local_dir}")
        return 0

    print(f"Local:  {local_dir}")
    print(f"S3:     s3://{args.bucket}/{prefix}")
    print("Listing existing objects...")
    existing = _list_existing_keys(s3, args.bucket, prefix)
    print(f"Existing objects under prefix: {len(existing)}")

    to_upload: list[tuple[Path, str]] = []
    for p in files:
        key = prefix + p.name
        if key in existing:
            continue
        to_upload.append((p, key))

    print(f"Files total: {len(files)}")
    print(f"To upload:   {len(to_upload)}")
    if args.dry_run:
        for p, key in to_upload[:200]:
            print(f"DRY-RUN upload: {p.name} -> s3://{args.bucket}/{key}")
        if len(to_upload) > 200:
            print(f"... and {len(to_upload) - 200} more")
        return 0

    workers = max(1, int(args.workers))
    workers = min(workers, len(to_upload)) if to_upload else 1

    uploaded = 0
    failed = 0

    def upload_one(item: tuple[Path, str]) -> None:
        src, key = item
        extra = {"ContentType": _guess_content_type(src)}
        s3.upload_file(str(src), args.bucket, key, ExtraArgs=extra)

    print(f"Uploading with workers={workers} ...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(upload_one, item): item for item in to_upload}
        for fut in as_completed(futures):
            src, key = futures[fut]
            try:
                fut.result()
                uploaded += 1
            except Exception as e:
                failed += 1
                print(f"FAILED: {src.name} -> {key}: {e}", file=sys.stderr)

    print(f"Uploaded: {uploaded}")
    print(f"Failed:   {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
