"""Corrective RED: disjoint rank intervals must not prove metadata capacity."""

from __future__ import annotations

import unittest

from fsbdd_stage0.storage_capacity import StorageHarnessError, _metadata_summary
from tests.storage.test_stage0_storage_capacity import _config, _synthetic_roles


class TestS0B03NonsynchronousMetadataSurrogate(unittest.TestCase):
    def test_rank_local_intervals_without_coordinator_timing_are_rejected(self) -> None:
        roles = _synthetic_roles(_config())
        with self.assertRaisesRegex(StorageHarnessError, "coordinator interval"):
            _metadata_summary(roles, _config())


if __name__ == "__main__":
    unittest.main()
