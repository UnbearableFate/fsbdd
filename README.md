Filesystem-based Decoupled DiLoCo
=================================

The source tree separates the production protocol from experiment support:

- `fsbdd.diloco.common`: canonical identities and structured runtime logging.
- `fsbdd.diloco.model`: logical-layer ownership, fragmentation, immutable model
  assets, and frozen evaluation snapshots.
- `fsbdd.diloco.protocol`: POSIX publication, proposals, global state, and
  atomic state-plus-consumption commits.
- `fsbdd.diloco.learner`: independent learner execution, fragment publication,
  and latest-complete-fragment adoption.
- `fsbdd.diloco.syncer`: readiness, bounded streaming merge, and Profile A
  progress/execution.
- `fsbdd.auxiliary`: configuration/evidence contracts, Stage 0 tools, Stage 1
  stress workloads, offline analysis, and closure-run harnesses.

Production modules must not import `fsbdd.auxiliary`. Auxiliary code may use
the production package to build tests, analyzers, and PBS entry points.
