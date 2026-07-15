from __future__ import annotations

import contextlib
import hashlib
import sys
from collections.abc import Iterator

import numpy as np

from fsbdd.global_state import FragmentStateDescriptor
from fsbdd.syncer_merge import (
    ContributionFact,
    FragmentMergeRequest,
    FragmentOuterState,
    OuterSGDPolicy,
    execute_numpy_streaming_fragment_update,
)


def _payload(values: list[float]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


class _Source:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    @contextlib.contextmanager
    def open_payload(self, _contribution: ContributionFact) -> Iterator[bytes]:
        yield self.payload


def main() -> int:
    current = _payload([2.0, 4.0])
    local = _payload([1.0, 3.0])
    content = hashlib.sha256(b"current").hexdigest()
    fact = ContributionFact(
        proposal_id="proposal-0",
        content_identity=hashlib.sha256(b"proposal").hexdigest(),
        learner_id="learner-00",
        base_version=0,
        base_content_identity=content,
        processed_tokens=10,
        staleness=0,
        normalized_weight=1.0,
        parameters_sha256=hashlib.sha256(local).hexdigest(),
        payload_bytes=len(local),
    )
    request = FragmentMergeRequest(
        descriptor=FragmentStateDescriptor(
            0,
            hashlib.sha256(b"fragment").hexdigest(),
            "float32",
            (2,),
            ("parameter-0",),
        ),
        fragment_map_identity=hashlib.sha256(b"map").hexdigest(),
        current_version=0,
        current_content_identity=content,
        current_parameters=current,
        outer_state=FragmentOuterState(0),
        selection_identity=hashlib.sha256(b"selection").hexdigest(),
        contributions=(fact,),
    )
    result = execute_numpy_streaming_fragment_update(
        request,
        _Source(local),
        OuterSGDPolicy(0.1, 0.9, True),
    )
    if "torch" in sys.modules or result.outer_state.update_count != 1:
        return 1
    print("numpy-syncer-probe: pass; torch_module_imported=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
