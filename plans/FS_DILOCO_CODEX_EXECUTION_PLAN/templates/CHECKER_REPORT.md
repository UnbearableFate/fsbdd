# Checker Report：<LOOP-ID>/<RUN-ID>

## Verdict

Phase A：`NOT_REQUIRED | ADMISSIBLE | BLOCKED_PRECHECK`

Phase B：`PASS | PASS_WITH_FOLLOWUPS | BLOCKED`

## 独立输入

- source hashes：
- loop card：
- code commit/diff：
- resolved config：
- RED/GREEN/HARDEN evidence：
- 9N package（适用）：
- requirement/measurement/evidence matrix：
- package validator summary：

## Phase A：昂贵运行前检查

- requirement/acceptance 是否完整：
- 反例是否覆盖主要错误实现：
- authoritative observation 与 aggregation 是否有效：
- run/evidence contract 是否 admissible：
- 允许升级到的最高资源等级：

## 重新构建的目标

- 单一目标：
- 受影响不变量：
- Acceptance：
- 用户补充门禁：

## RED 有效性

- 错误实现/缺失实现是否失败：
- 失败是否对应需求：
- 是否存在偶然 timeout/脆弱断言：

## 代码与协议检查

- scope：
- data plane：
- bounded state/discovery：
- atomic publication：
- base/consumption：
- numerical reference：
- config/identity：
- mutable-container bounded-state inventory：
- resume code/config/generator/formula identity：

## Maker 未列出的反例

1.
2.

## 运行证据

- node/role：
- loss：
- runtime：
- global cycles/local steps：
- fault/perf overlay：
- raw evidence 可复算：
- timestamp source / scheduler identity：
- final package validator：

## Acceptance 判定

| ID | 证据 | 结论 |
|---|---|---|
| | | |

## Follow-ups

只列不影响当前关闭的项目；否则 verdict 必须 BLOCKED。

## 签署

- Checker session/date：
- Checker context ID / Maker chat inherited=false：
- checked commit：
- report hash：
