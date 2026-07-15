from __future__ import annotations

import csv
import re
from pathlib import Path


class MatrixError(ValueError):
    pass


_ID = re.compile(r"\b(?:[A-Z][A-Z0-9]*-)+[0-9]+\b")
_COLUMNS = {
    "id",
    "assertion",
    "counterexample",
    "target_api",
    "authoritative_observation",
    "aggregation",
    "evidence_path",
    "close_condition",
}


def authority_ids(loop_card: Path) -> set[str]:
    text = loop_card.read_text(encoding="utf-8")
    authority_lines = [
        line for line in text.splitlines() if "规范条款" in line or "Acceptance" in line
    ]
    found = set(_ID.findall("\n".join(authority_lines)))
    if not found:
        raise MatrixError("no authority IDs found in loop card")
    return found


def validate_matrix(loop_card: Path, matrix_path: Path) -> dict[str, object]:
    expected = authority_ids(loop_card)
    with matrix_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != _COLUMNS:
            raise MatrixError("requirement matrix columns do not match schema")
        rows = list(reader)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise MatrixError("duplicate requirement IDs")
    missing = expected - set(ids)
    extra = set(ids) - expected
    if missing or extra:
        raise MatrixError(
            f"requirement set mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )
    for row in rows:
        empty = sorted(column for column in _COLUMNS if not row[column].strip())
        if empty:
            raise MatrixError(f"empty fields for {row['id']}: {empty}")
    return {
        "status": "complete",
        "expected_ids": sorted(expected),
        "row_count": len(rows),
    }
