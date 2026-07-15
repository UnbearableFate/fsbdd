from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsbdd.auxiliary.experiment_failure import (
    ExperimentFailureError,
    capture_failure,
)


def test_capture_failure_writes_required_review_fields(tmp_path: Path) -> None:
    output_root = (tmp_path / "run").resolve()
    path, record = capture_failure(
        output_root=output_root,
        experiment_id="s1-13-test",
        exit_code=1,
        line_number=42,
        command="python -m failing_workload",
        job_id="123.opbs",
        role_id="8",
        code_commit="a" * 40,
        timestamp_utc="2026-07-15T16:33:29Z",
    )

    assert path.parent == output_root / "failures"
    assert json.loads(path.read_text(encoding="utf-8")) == record
    assert record["analysis_status"] == "pending_review"
    assert record["failure"] == {
        "summary": "experiment command exited with a nonzero status",
        "exit_code": 1,
        "line_number": 42,
        "command": "python -m failing_workload",
    }
    assert record["cause"]["root_cause"] is None
    assert record["expected_solution"]["status"] == "review_required"


@pytest.mark.parametrize(
    ("exit_code", "line_number"),
    ((0, 1), (-1, 1), (1, 0), (1, -1), (True, 1), (1, True)),
)
def test_capture_failure_rejects_invalid_failure_coordinates(
    tmp_path: Path, exit_code: int, line_number: int
) -> None:
    with pytest.raises(ExperimentFailureError):
        capture_failure(
            output_root=(tmp_path / "run").resolve(),
            experiment_id="s1-13-test",
            exit_code=exit_code,
            line_number=line_number,
            command="false",
            job_id="123.opbs",
        )


@pytest.mark.parametrize(
    ("timestamp_utc", "code_commit"),
    (
        ("not-a-timeZ", "a" * 40),
        ("2026-07-15T16:33:29+09:00", "a" * 40),
        ("2026-07-15T16:33:29Z", "not-a-commit"),
    ),
)
def test_capture_failure_rejects_invalid_run_identity(
    tmp_path: Path, timestamp_utc: str, code_commit: str
) -> None:
    with pytest.raises(ExperimentFailureError):
        capture_failure(
            output_root=(tmp_path / "run").resolve(),
            experiment_id="s1-13-test",
            exit_code=1,
            line_number=1,
            command="false",
            job_id="123.opbs",
            code_commit=code_commit,
            timestamp_utc=timestamp_utc,
        )
