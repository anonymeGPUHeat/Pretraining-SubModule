#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
from collections import defaultdict, Counter
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


logger = logging.getLogger("search_ptx_patterns")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PATTERNS = {
    "dot_tex": re.compile(r"\.tex", re.IGNORECASE),
    "dot_atom": re.compile(r"\.atom", re.IGNORECASE),
    "wmma.mma": re.compile(r"\bwmma\.mma\b"),
    "wgmma.mma_async": re.compile(r"\bwgmma\.mma_async\b"),
    "mma": re.compile(r"\bmma\b"),
}


RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
RE_LONG_PUNCT = re.compile(r"[^A-Za-z0-9\s]{3,}")
RE_NONASCII = re.compile(r"[^\x00-\x7f]")
RE_DOUBLE_COLON = re.compile(r"::")


def find_files(root, exts=None):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not exts or os.path.splitext(fn)[1].lower() in exts:
                yield os.path.join(dirpath, fn)


def scan_file(path):
    results = defaultdict(list)
    logger.debug("Scanning file: %s", path)
    try:
        with open(path, "r", encoding="utf-8", errors="backslashreplace") as f:
            text = f.read()
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        results["read_error"].append(str(e))
        return results

    for name, rx in PATTERNS.items():
        for m in rx.finditer(text):
            results[name].append(m.group(0))

    for m in RE_CONTROL.finditer(text):
        results["control_chars"].append(m.group(0))
    for m in RE_LONG_PUNCT.finditer(text):
        results["long_punct_runs"].append(m.group(0))
    for m in RE_NONASCII.finditer(text):
        results["non_ascii"].append(m.group(0))
    for m in RE_DOUBLE_COLON.finditer(text):
        results["double_colon"].append(m.group(0))

    return results


def summarize(root, exts=None, out_json=None):
    stats = {
        "scanned_files": 0,
        "files_with_matches": 0,
        "pattern_counts": Counter(),
        "files": defaultdict(lambda: defaultdict(int)),
        "unusual_examples": defaultdict(list),
    }

    for idx, path in enumerate(find_files(root, exts=exts), start=1):
        stats["scanned_files"] += 1
        if idx % 10 == 0:
            logger.info("Scanned %d files...", idx)
        res = scan_file(path)
        matched = False
        for k, vals in res.items():
            if vals:
                matched = True
                stats["pattern_counts"][k] += len(vals)
                stats["files"][k][path] += len(vals)
                existing_examples = stats["unusual_examples"][k]
                if len(existing_examples) < 5:
                    new_examples = []
                    for v in vals:
                        if v not in existing_examples and v not in new_examples:
                            new_examples.append(v)
                    stats["unusual_examples"][k].extend(new_examples[:5 - len(existing_examples)])

        if matched:
            stats["files_with_matches"] += 1

    out = {
        "scanned_files": stats["scanned_files"],
        "files_with_matches": stats["files_with_matches"],
        "pattern_counts": dict(stats["pattern_counts"]),
        "files": {k: dict(v) for k, v in stats["files"].items()},
        "unusual_examples": {k: v for k, v in stats["unusual_examples"].items()},
    }

    if out_json:
        try:
            with open(out_json, "w", encoding="utf-8") as of:
                json.dump(out, of, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to write JSON: {e}")

    return out


def main():
    p = argparse.ArgumentParser(description="Search PTX files for patterns and unusual tokens")
    p.add_argument("root", nargs="?", default=os.path.join("data", "raw"),
                   help="Root folder to search (default: data/raw relative to project root)")
    p.add_argument("--ext", default=".ptx", help="Comma-separated extensions to search (default: .ptx)")
    p.add_argument("--out", default=os.path.join("scripts", "ptx_pattern_results.json"),
                   help="Output JSON file (path relative to project root if not absolute)")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose (debug) logging")
    p.add_argument("--logfile", default=None, help="Optional log file path")
    args = p.parse_args()

    exts = None
    if args.ext:
        exts = tuple(x.strip().lower() if x.strip().startswith('.') else '.' + x.strip().lower() for x in args.ext.split(','))

    log_level = logging.DEBUG if args.verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    if args.logfile:
        handlers.append(logging.FileHandler(args.logfile))
    logging.basicConfig(level=log_level, handlers=handlers, format="%(asctime)s %(levelname)s %(message)s")

    if os.path.isabs(args.root):
        root_path = args.root
    else:
        root_path = os.path.abspath(os.path.join(PROJECT_ROOT, args.root))

    out_json = args.out if os.path.isabs(args.out) else os.path.abspath(os.path.join(PROJECT_ROOT, args.out))
    out_dir = os.path.dirname(out_json)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    logger.info("Scanning '%s' for extensions: %s", root_path, exts or 'all files')
    res = summarize(root_path, exts=exts, out_json=out_json)
    logger.info("Summary:")
    logger.info(" Scanned files: %d", res["scanned_files"])
    logger.info(" Files with matches: %d", res["files_with_matches"])
    logger.info(" Pattern counts:")
    for k, v in sorted(res["pattern_counts"].items(), key=lambda x: -x[1]):
        logger.info("  - %s: %d", k, v)

    logger.info("Files per pattern (showing up to 10 files each):")
    for k, files in res["files"].items():
        logger.info("  %s: %d files", k, len(files))
        for i, (fn, cnt) in enumerate(sorted(files.items(), key=lambda x: -x[1])):
            if i >= 10:
                break
            logger.info("    %4d  %s", cnt, fn)

    logger.info("JSON results written to: %s", out_json)


if __name__ == "__main__":
    main()


"""
Search PTX files under data/raw for specific and unusual patterns so i can see what needs to be cleaned

Finds occurrences of:
- ".tex"
- ".atom"
- tensor patterns: "wmma.mma", "mma", "wgmma.mma_async"
- unusual patterns: control chars, long runs of punctuation, non-ASCII, tokens containing '::' or unusual symbols

Outputs a JSON summary and a human-readable report.
"""