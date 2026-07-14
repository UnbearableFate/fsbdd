# Stage 0-A Simulation Recommendation

## Scope

This report is a parameter-selection and later-measurement anchor. It is not
runtime correctness, model-quality, or a reason to cancel Stage 4.

## Matrix and uncertainty

- Complete SIM-03 extended matrix: `7776` / `7776` raw rows.
- Three seeds per profile; reported uncertainty is sample standard deviation and a normal 95% CI half-width.
- Full seed-level CSV and `2592` profile aggregates remain in the immutable evidence run `20260714T145700Z-green3-resume-n7b4a-14126dde2f00`.
- Recovery is paired `accepted_token_efficiency(S_max=1) - accepted_token_efficiency(S_max=0)` in fraction units; percentage points are `100 × fraction`.

## Stage 1 Profile A recommendation

Freeze the fresh reference at `M=8`, `Q=M`, `Q_fresh=M`,
`S_max=0`, `grace=0`, `lambda_s=1.0`, `F=4`, and evenly
staggered `H=50` steps. The sweep evaluates `Q_fresh=1`, which is behaviorally
equivalent in this fresh-only `Q=M` arm because every eligible contributor is fresh.
Stage 0-B must still validate or mitigate the storage-dependent H/visibility budget.

## Stage 4 Profile B prediction anchor

The preregistered ranking selects `M=8`, `Q=4`
(`Q/M=0.5`), `Q_fresh=1`, `S_max=1`,
`lambda_s=1.0`, grace `0.1H`,
`F=4`, and `H=50` steps. The validation
anchor injects constant heterogeneity `2.0×` and
visibility `1.0s` (`delay/H=0.020000`).
Here `H` in seconds uses the simulator's explicit nominal-fastest
`1.0s/step` reference; real training must
recompute the ratio from its measured step time.

- predicted recovery: `-0.0446` percentage points, 95% CI half-width `0.0000` pp;
- stale accepted-token rate: `5.3108%` (95% CI half-width `0.0000` pp);
- accepted-token efficiency: `33.4834%`;
- paper classification: `ablation_or_discussion`.

Stage 4 remains mandatory. Its measured stale acceptance and recovery must be
reported against this exact anchor and the matched `S_max=0` control.

## Stage 5 delay recommendation

Retain the preregistered visibility points `0/1/5/30s`, always reporting both
seconds and `visibility delay / H`. These span the observed no-delay reference,
low-delay operating region, intermediate degradation, and high-delay stress point.
Do not infer storage capability from the simulation; use Stage 0-B measurements.
