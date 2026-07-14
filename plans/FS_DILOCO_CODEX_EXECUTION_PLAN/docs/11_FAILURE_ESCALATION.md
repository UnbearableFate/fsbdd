# 11. 失败分类、阻塞与升级

## 11.1 分类

| 类别 | 例子 | 第一动作 |
|---|---|---|
| SPEC | 条款冲突、缺少定义 | 写 ADR/问题，不猜 |
| CODE | test/exception/wrong result | 最小复现 |
| NUMERIC | oracle mismatch、NaN | 缩小为标量/小模型 |
| CONCURRENCY | race、partial visibility | 固定 schedule/fault point |
| STORAGE | visibility/atomicity/quota | Stage 0-B 复测/缓解 |
| ENV | module/package/cache | 记录环境，使用 skill |
| PBS | allocation/launcher/node | 静态检查+最小作业 |
| PERF | slow/stall/queue growth | 分解 profile |
| EVIDENCE | 日志/manifest缺失 | 修 harness，重跑 |
| RESEARCH | H3/H1 不成立 | 保留负面结果，不改口径 |

## 11.2 BLOCKED 报告

必须包含：

- loop/commit/config/run IDs；
- 预期与实际；
- 最小复现命令；
- 原始错误/trace；
- 已排除原因；
- 分类；
- 影响的 requirement/acceptance；
- 是否污染已有证据；
- 下一条最小动作；
- 是否需要用户/管理员/资源授权。

## 11.3 站点问题

不得绕过 Miyabi 规则。需要管理员时提供：

- PBS job ID/time；
- host/queue/group；
- minimal script；
- module list；
- error output；
- 是否可重复；
- 不包含私密 token/路径内容。

## 11.4 规格升级

若正确实现无法满足 acceptance：

1. 保留失败；
2. 验证是否环境/参数；
3. Checker 复核；
4. 写 ADR，提出至少两个选项；
5. 分析研究 claim 影响；
6. 用户裁决；
7. 修改规范；
8. 重新开放受影响 loops。

## 11.5 负面研究结果

H3、性能阈值等未成立不是工程失败。前提是：

- 实现正确；
- matched 口径正确；
- 证据完整；
- 未通过优化/阈值事后选择掩盖。

将结果标为 `RESEARCH_NEGATIVE_RESULT`，实现/测试仍可 PASS，研究 claim 按源计划回退。
