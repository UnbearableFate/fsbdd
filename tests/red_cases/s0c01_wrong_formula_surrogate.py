"""Deliberately wrong stale displacement used only to prove RED sensitivity."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "oracle_golden_cases.json"


class TestWrongCurrentMinusLocalSurrogate(unittest.TestCase):
    def test_inv_07__wrong_current_minus_local_fails_numerically(self) -> None:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        case = next(
            item
            for item in fixture["pseudo_gradient_cases"]
            if item["id"] == "stale-s1-counterexample"
        )
        wrong = [
            current - local
            for current, local in zip(case["current"], case["local"], strict=True)
        ]
        self.assertEqual(
            wrong,
            case["expected"],
            msg=(
                "INV-07 violation: G_current - L_stale differs from the required "
                "G_base - L_stale"
            ),
        )


if __name__ == "__main__":
    unittest.main()
