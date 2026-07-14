# 里程碑

| 里程碑 | 触发 | 必需证据 |
|---|---|---|
| M0 Oracle Ready | S0C-02 | math/consumption/outer transitions |
| M1 Storage Qualified | S0B-03 | visibility/atomicity/metadata/bandwidth |
| M2 Stage 0 Closed | 全Stage0 | simulator defaults + Stage0 checker |
| M3 Protocol Core | S1-11 | proposal/global commit closed loop |
| M4 Base Training | S1-13 | CAP-BASE-TRAINING + BASELINE-9N-v1 |
| M5 Performance Ready | S2-05 | 95% goodput/2% pause/slow-FS bounded |
| M6 Crash Consistent | S3-04 | REC matrix |
| M7 Stale Mechanism | S4-05 | Profile B stale acceptance/sim alignment |
| M8 Stage 4 Closed | S4-06 | matched-token H3 and fallback decision |

每个里程碑写 Stage/loop checkpoint，包含 branch/commit、配置、证据和Checker verdict。
