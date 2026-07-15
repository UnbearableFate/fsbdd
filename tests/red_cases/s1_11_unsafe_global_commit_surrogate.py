from __future__ import annotations

import dataclasses
import unittest


@dataclasses.dataclass
class UnsafeSplitAuthority:
    parameters: bytes = b"parameters-v0"
    outer_state: bytes = b"outer-v0"
    version: int = 0
    consumed_sequence: int = -1
    consumed_base: int = -1

    def publish_split_then_interrupt(self) -> None:
        self.parameters = b"parameters-v1"
        raise RuntimeError("interrupted between independent latest files")

    def consume_before_visibility_then_interrupt(self) -> None:
        self.consumed_sequence = 7
        self.consumed_base = 0
        raise RuntimeError("interrupted before global current visibility")

    def publish_requested_version(self, version: int, parameters: bytes) -> None:
        self.version = version
        self.parameters = parameters

    def proposal_is_eligible(self, sequence: int, base_version: int) -> bool:
        return sequence > self.consumed_sequence and base_version > self.consumed_base


class S111SemanticRed(unittest.TestCase):
    def test_inv_03__split_publication_exposes_mixed_authority(self) -> None:
        authority = UnsafeSplitAuthority()
        old = (authority.parameters, authority.outer_state, authority.version)
        new = (b"parameters-v1", b"outer-v1", 1)
        with self.assertRaises(RuntimeError):
            authority.publish_split_then_interrupt()
        observed = (authority.parameters, authority.outer_state, authority.version)
        self.assertIn(
            observed,
            (old, new),
            "INV-03/GLOBAL-02 require one old-or-new compound current authority",
        )

    def test_inv_05__failed_publication_must_not_consume(self) -> None:
        authority = UnsafeSplitAuthority()
        with self.assertRaises(RuntimeError):
            authority.consume_before_visibility_then_interrupt()
        self.assertTrue(
            authority.proposal_is_eligible(7, 0),
            "INV-05/SYNC-08 require failed publication to preserve eligibility",
        )

    def test_global_03__version_jump_must_fail(self) -> None:
        authority = UnsafeSplitAuthority()
        authority.publish_requested_version(2, b"skipped-v1")
        self.assertEqual(
            authority.version,
            1,
            "GLOBAL-03 requires every successful version to advance exactly one",
        )

    def test_global_03__same_version_different_content_must_fail(self) -> None:
        authority = UnsafeSplitAuthority()
        authority.publish_requested_version(1, b"content-a")
        first = authority.parameters
        authority.publish_requested_version(1, b"content-b")
        self.assertEqual(
            authority.parameters,
            first,
            "GLOBAL-03 forbids overwriting one version with different content",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
