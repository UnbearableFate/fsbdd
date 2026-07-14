# S1-02 Fragment Map Contract

The production API accepts only `1 < F < L`. A separate explicitly labeled
oracle mode permits only `F = 1`; `F >= L` is never accepted. Each fragment is
an immutable contiguous interval of complete logical layers from the S1-01
registry, and validation proves that the flattened fragment identities equal
the registry parameter identities exactly once.

Optimization uses integer synchronization bytes. The objective tuple is:

1. minimum possible maximum fragment bytes;
2. minimum `sum(abs(F * fragment_bytes - total_bytes))` under that cap;
3. lexicographically earliest exclusive cut-index tuple.

The first stage is constrained minimax dynamic programming. Once its cap is
fixed, a second dynamic program minimizes the additive exact deviation and
the front-cut tie-break. No floating-point value participates in selection.
The canonical map digest covers the registry digest, ordered layer summaries,
parameter identities, misc and tied ownership, fragment intervals and bytes,
the exact objective, and balance statistics. The report is diagnostic; the
frozen `FragmentMap` and digest are runtime authority, and there is no API for
runtime repartitioning.
