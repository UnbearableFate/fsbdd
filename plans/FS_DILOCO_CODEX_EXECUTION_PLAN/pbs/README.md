# Miyabi PBS 模板

这些文件是 Codex 生成项目脚本时的受控模板，不是无需修改即可提交的站点脚本。

强制步骤：

1. 使用 `miyabi-development` skill；login/compute 分工与执行位置一律按 skill 的
   host routing，本包不另行规定；
2. 从当前账户/项目发现 queue、literal `group_list`、modules、node shape、walltime；
3. 填充所有 `<...>`，包括已验证的 1 CPU + 8 GPU coallocated resource string 与
   independent exclusive-host placement；
4. 记录 script SHA、skill commit和module list。

文件：

- `coallocated_9node.pbs.template`：每-loop门禁的首选；一个9节点allocation，以MPI/PBS仅启动角色。
- `independent_bootstrap.pbs.template`：在 compute node 创建单次 run 的 bootstrap。
- `independent_syncer.pbs.template`、`independent_learner.pbs.template`：Stage1场景真实性、Stage3恢复；roles 使用原子 host claim 拒绝重复物理 host。
- `submit_independent_8l1s.sh.template`：生成唯一 run ID，保留独占 run/evidence
  目录，提交 bootstrap，并以 `afterok` 依赖提交一个 syncer 和八个 learners。

每次 run 使用 `RUN_ROOT_BASE/$RUN_ID`，不得把 `RUN_ROOT_BASE` 本身直接作为协议根。
evidence 目录必须 fail-if-exists，不能用 `mkdir -p` 覆盖旧 run。

应用层不得调用MPI collective、torch.distributed、NCCL或RPC。共享FS仍是唯一算法数据面。
