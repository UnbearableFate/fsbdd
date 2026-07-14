# 计划级配置模板

这些 YAML 描述必须进入最终项目 config schema 的字段与门禁含义，不保证可直接被尚未实现的 CLI 读取。

- `miyabi_8l1s_50x10.yaml`：用户补充的每-loop真实门禁。
- `profile_a_fresh.yaml`：Stage 1 fresh reference。
- `profile_b_stale.yaml`：Stage 4 stale-aware。
- `stage0_simulation_scan.yaml`：SIM-03。
- `stage0_storage_benchmark.yaml`：BENCH-01..04。

Codex 实现配置系统时：

1. 用 typed schema 校验；
2. 合并后输出 resolved config；
3. 在提交前把所有 `null`/占位符填完；
4. 计算 SHA-256；
5. 所有角色校验同一 digest；
6. 不允许 runtime 修改算法字段。
