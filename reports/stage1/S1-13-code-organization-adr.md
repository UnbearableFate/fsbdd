# S1-13 code-organization ADR

Status: accepted by explicit user direction on 2026-07-16.

## Context

The implementation and all Stage 0/Stage 1 experiment support previously
shared one flat `fsbdd` namespace, while the Stage 0 simulator occupied a
second top-level package.  That layout obscured the production dependency
boundary and made it easy for filesystem protocol code to acquire evidence or
experiment-harness dependencies.

The exact five-node long-profile smoke at commit `0023c3d` completed before
this reorganization.  It preserved a natural frozen-budget failure at global
cycle 35 of 59 in
`runtime_runs/S1-13/long-smoke-20260715T161208Z-0023c3d`; no later job was
submitted before this change.

## Decision

Production code lives below `fsbdd.diloco`, divided into `common`, `model`,
`protocol`, `learner`, and `syncer`.  Configuration/evidence contracts,
analysis, Stage 0 utilities, stress workloads, and Stage 1 run orchestration
live below `fsbdd.auxiliary`.

The dependency direction is one way: auxiliary code may import production
code, while production code may not import auxiliary code.  PBS entry points,
tests, and static source audits use the new authoritative module paths.  This
is a source-layout change, not an algorithm, protocol, topology, or threshold
change.

## Validation and rollback

An architecture test enforces the dependency boundary.  The complete Stage 0
and Stage 1 suites, exact compute-node tests, and runtime smokes must pass on a
clean committed worktree before any formal run.  Rollback is the single
organization commit if import migration exposes an unresolvable external
contract; no evidence directory is rewritten or reused.
