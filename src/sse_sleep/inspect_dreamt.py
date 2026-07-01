"""Inspect DreamT data files before implementing a dataset-specific loader.

This module intentionally uses only Python's standard library so it can run
before the ML dependencies are installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .labels import map_label_5
from .schema import channel_alias_index, feature_catalog_dicts


TEXT_EXTENSIONS = {".csv", ".tsv", ".txt"}
KNOWN_DATA_EXTENSIONS = TEXT_EXTENSIONS | {".edf", ".mat", ".h5", ".hdf5", ".parquet", ".npy", ".npz"}
COPY_PREFIX_PATTERNS = (
    re.compile(r"^copy of\s+", re.IGNORECASE),
    re.compile(r"^사본\s+"),
)
COPY_SUFFIX_PATTERNS = (
    re.compile(r"\s*-\s*copy$", re.IGNORECASE),
    re.compile(r"\s*-\s*사본$"),
    re.compile(r"\s+copy$", re.IGNORECASE),
    re.compile(r"\s+사본$"),
    re.compile(r"의\s*사본$"),
    re.compile(r"\s*\([0-9]+\)$"),
)


@dataclass
class ColumnSummary:
    name: str
    inferred_channel: str | None
    non_empty_in_sample: int
    unique_values_in_sample: int
    possible_stage_labels: dict[str, int]


@dataclass
class FileSummary:
    path: str
    extension: str
    size_bytes: int
    kind: str
    delimiter: str | None = None
    columns: list[ColumnSummary] | None = None
    sample_rows_read: int = 0
    note: str | None = None


def normalize_column(name: str) -> str:
    text = name.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def infer_channel(column_name: str, aliases: dict[str, str]) -> str | None:
    normalized = normalize_column(column_name)
    if normalized in aliases:
        return aliases[normalized]
    for alias, canonical in aliases.items():
        if len(alias) > 2 and alias in normalized:
            return canonical
    return None


def copy_normalized_stem(stem: str) -> str:
    """Return a stem with common Google Drive/Finder copy markers removed."""
    normalized = stem.strip()
    changed = True
    while changed:
        changed = False
        for pattern in COPY_PREFIX_PATTERNS:
            updated = pattern.sub("", normalized).strip()
            if updated != normalized:
                normalized = updated
                changed = True
        for pattern in COPY_SUFFIX_PATTERNS:
            updated = pattern.sub("", normalized).strip()
            if updated != normalized:
                normalized = updated
                changed = True
    return normalized


def is_copy_like_file(path: Path) -> bool:
    """Detect common duplicate-copy file names without looking at file contents."""
    return copy_normalized_stem(path.stem) != path.stem.strip()


def duplicate_group_key(root: Path, path: Path) -> tuple[str, str, str]:
    """Group original/copy files by relative parent, normalized stem, and extension."""
    relative_parent = str(path.parent.relative_to(root))
    return (relative_parent, copy_normalized_stem(path.stem).lower(), path.suffix.lower())


def sniff_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        return ","


def inspect_text_table(path: Path, sample_rows: int) -> FileSummary:
    delimiter = sniff_delimiter(path)
    aliases = channel_alias_index()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        fieldnames = list(reader.fieldnames or [])
        counters = {name: Counter() for name in fieldnames}
        non_empty = Counter()
        rows_read = 0
        for row in reader:
            rows_read += 1
            for name in fieldnames:
                value = (row.get(name) or "").strip()
                if value:
                    non_empty[name] += 1
                    if len(counters[name]) < 100:
                        counters[name][value] += 1
            if rows_read >= sample_rows:
                break

    columns: list[ColumnSummary] = []
    for name in fieldnames:
        stage_counter: Counter[str] = Counter()
        for value, count in counters[name].items():
            mapped = map_label_5(value)
            if mapped.canonical is not None:
                stage_counter[mapped.canonical] += count
        columns.append(
            ColumnSummary(
                name=name,
                inferred_channel=infer_channel(name, aliases),
                non_empty_in_sample=non_empty[name],
                unique_values_in_sample=len(counters[name]),
                possible_stage_labels=dict(stage_counter),
            )
        )

    return FileSummary(
        path=str(path),
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size,
        kind="text_table",
        delimiter=delimiter,
        columns=columns,
        sample_rows_read=rows_read,
    )


def all_data_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in KNOWN_DATA_EXTENSIONS
    ]


def select_data_files(root: Path, include_copy_files: bool) -> tuple[list[Path], list[Path]]:
    """Select data files, skipping copy-like duplicates when an original exists."""
    files = all_data_files(root)
    if include_copy_files:
        return files, []

    grouped: dict[tuple[str, str, str], list[Path]] = {}
    for path in files:
        grouped.setdefault(duplicate_group_key(root, path), []).append(path)

    selected: list[Path] = []
    skipped: list[Path] = []
    for paths in grouped.values():
        originals = [path for path in paths if not is_copy_like_file(path)]
        if originals:
            selected.extend(originals)
            skipped.extend(path for path in paths if is_copy_like_file(path))
        else:
            selected.extend(paths)
    return sorted(selected), sorted(skipped)


def inspect_file(path: Path, sample_rows: int) -> FileSummary:
    extension = path.suffix.lower()
    if extension in TEXT_EXTENSIONS:
        return inspect_text_table(path, sample_rows)
    return FileSummary(
        path=str(path),
        extension=extension,
        size_bytes=path.stat().st_size,
        kind="binary_or_columnar",
        note="Install/read with format-specific loader to inspect channels and sampling rates.",
    )


def inspect_root(root: Path, sample_rows: int, include_copy_files: bool = False) -> dict[str, object]:
    selected_files, skipped_copy_files = select_data_files(root, include_copy_files)
    files = [inspect_file(path, sample_rows) for path in selected_files]
    return {
        "root": str(root),
        "include_copy_files": include_copy_files,
        "skipped_copy_file_count": len(skipped_copy_files),
        "skipped_copy_files": [str(path) for path in skipped_copy_files],
        "file_count": len(files),
        "files": [asdict(file_summary) for file_summary in files],
        "feature_catalog": feature_catalog_dicts(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DreamT dataset structure.")
    parser.add_argument("--root", type=Path, required=True, help="DreamT dataset root directory.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report output path.")
    parser.add_argument("--sample-rows", type=int, default=1000, help="Rows sampled per text table.")
    parser.add_argument(
        "--include-copy-files",
        action="store_true",
        help="Include files that look like Google Drive/Finder duplicate copies.",
    )
    args = parser.parse_args()

    if not args.root.exists():
        raise SystemExit(f"Dataset root does not exist: {args.root}")

    report = inspect_root(args.root, args.sample_rows, include_copy_files=args.include_copy_files)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
