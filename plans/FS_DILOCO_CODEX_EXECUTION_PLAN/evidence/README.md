# Evidence 目录规范

项目实施时使用：

```text
evidence/
  <loop-id>/
    <run-or-check-id>/
      manifest.yaml
      commands.log
      configs/
      env/
      stdout/
      stderr/
      tests/
      runtime/
      metrics/
      storage_snapshot/
      analysis/
      checker/
      checksums.sha256
```

规则：

- 每次失败/成功 run 使用新 ID；
- 原始文件不可原地改写；
- analysis 引用原始文件 hash；
- secrets 不进入 evidence；
- Stage closure 的证据不可 GC；
- 大 tensor payload 可不永久保留，但 metadata、identity、tree/size snapshot 与重算所需样本必须保留；
- tracked `$REPO_ROOT/evidence/indexes/<loop-id>.md` 是人类入口，raw run 中的
  `checksums.sha256` 是完整性入口。
- raw evidence 不提交 Git；index 必须记录 resolved `EVIDENCE_ROOT` 位置。
