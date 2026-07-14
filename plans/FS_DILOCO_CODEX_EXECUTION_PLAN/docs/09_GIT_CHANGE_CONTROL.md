# 09. Git 与变更控制

## 9.1 分支

首次启动时检测真实 integration branch，并写入 loop state：

```bash
git symbolic-ref --quiet --short refs/remotes/origin/HEAD
```

当前仓库为 `master`。以下所有“不 merge main”类规则均指检测到的 integration
branch，不能按分支名字面规避。

每个 loop：

```text
codex/<loop-id-lower>-<short-slug>
```

例如：

```text
codex/s1-05-proposal-protocol
```

可从前一已通过 loop 分支或用户指定 integration branch 创建。开始时记录 base commit。

## 9.2 Commit 边界

建议至少：

1. `test(<loop>): add red evidence for <id>`
2. `feat/fix(<loop>): implement minimal green`
3. `test(<loop>): harden boundaries`
4. `docs(<loop>): persist evidence and traceability`

调试期 WIP commit 可以存在；交付前可在自己的分支清理，但必须 `--force-with-lease`，不重写他人分支。

## 9.3 本地—Miyabi 同步

只通过 GitHub feature branch 同步 tracked source。每次切换环境记录：

```bash
git status --short --branch
git rev-parse HEAD
```

Miyabi：

```bash
git fetch origin
git switch <branch>
git pull --ff-only
```

不在 Miyabi 上形成未回传的长期 source 漂移。

## 9.4 Merge 权限

Codex 默认：

- 可以创建、commit、push feature branch；
- 可以准备 clean history；
- 不 merge 检测到的 integration branch；
- 不删除 branch；
- 报告 `ready to merge` 和证据。

Miyabi runtime/PBS/训练改动即使用户曾允许低风险自动 merge，也需该类改动的明确授权。

执行包、source spec 与根目录执行契约必须先进入 tracked baseline。若原 checkout
存在与本 loop 无关的 dirty state，在提交计划内文件后从该 commit 创建 clean
worktree；不得提交、恢复或覆盖用户的无关修改。

## 9.5 ADR 触发

以下必须 ADR：

- 修改算法公式/eligibility/selection；
- 改 publication/authority；
- 改 fragment ownership；
- 改 recovery semantics；
- 取消/放宽 acceptance；
- 换 merge/outer optimizer 默认；
- 引入网络数据面；
- 改 9N loss/runtime 阈值；
- 换 model/data baseline；
- 改 Python/Torch/HF pin 导致结果不兼容；
- 偏离建议代码边界且影响多个 loops。

## 9.6 Spec 修改

先提交 ADR 和失败证据，再修改 source spec。source 文件变更必须：

- 更新版本/变更记录；
- 更新 hash；
- 更新所有引用；
- 标出重新开放的 loops/acceptance；
- 经用户确认。

执行包生成文件不得反向偷偷覆盖 source spec。
