# Stage 1：Fresh-only 最小端到端系统

## 目标

以 `S_max=0` 运行同一套通用 stale-ready 机制，实现可数值验证的 FS-only fragment training 闭环。技术底座为 PyTorch + Hugging Face，自定义 learner loop。

## 进入条件

- Stage 0-B 通过或缓解 profile 已冻结；
- Stage 0-C oracle 全部通过；
- Stage 0-A 默认参数建议可用；
- Miyabi `miyabi-development` skill 已安装/记录。

## Loop 顺序

| Loop | 内容 |
|---|---|
| S1-00 | 项目 skeleton、typed config、identity、manifest |
| S1-01 | HF model logical-layer registry |
| S1-02 | layer-aligned deterministic fragment map |
| S1-03 | STOR-01 interface 与 payload-first/visibility-last |
| S1-04 | bootstrap、per-fragment current/base bounded state |
| S1-05 | proposal schema、latest-wins、eligibility、consumption metadata |
| S1-06 | Torch/HF learner inner training core |
| S1-07 | H/offset snapshot 与 proposal publish |
| S1-08 | latest-only adoption、mixed-version、inner moments |
| S1-09 | syncer readiness、quorum、grace、公平调度 |
| S1-10 | base-relative math、direct merge、outer optimizer、streaming reduction |
| S1-11 | atomic global commit 与 consumed frontier |
| S1-12 | Profile A 数值 E2E 与 frozen evaluation snapshot |
| S1-13 | 基础系统关闭；首次 Miyabi 8+1/50×10 baseline |

## 技术约束

- 160M 级 decoder-only HF model 作为固定门禁模型；
- custom training loop；
- learner 不组成 DDP world；
- FS 是唯一算法数据面；
- `S_max=0` 只是 config；
- proposal 从第一天携带 base version/content identity/tokens/steps；
- weighting 从第一天使用统一函数；
- base window 从第一天按 `S_max+1`；
- steady-state 不传完整模型。

## 验收覆盖

- A-FRAG-01..03
- A-PROP-01..03
- A-GLOBAL-01..02
- A-LEARN-01..03
- A-PERF-01
- A-ALG-01
- A-EVAL-01

另需完成 source 规定的 160M、M=4、真实 FS、≥1B token run。`S1-13` 的 9 节点 50×10 不替代它。

## 基础能力标记

只有全部 Stage 1 acceptance、M=4/160M/真实 FS/≥1B-token long run、首次 9N gate
和 Stage 1 Checker 都通过，`S1-13` 才能标记 PASS 并写入：

```yaml
capability:
  id: CAP-BASE-TRAINING
  status: active
  baseline_9n_run_id: <run>
```

因此 `S2-01` 对 `S1-13` 的依赖同时也是 Stage 1 closure gate；不得先把 capability
标为 active、再补 long run。从此 Stage 2–4 每个实现 loop 都必须执行新的 9 节点门禁。

## Stage 关闭条件

- 所有 Stage 1 acceptance 通过；
- Profile A 单次 L2 error ≤1e-6，50 updates 无漂移放大；
- 真实 M=4 1B-token run loss 合理；
- 首次 8+1/50×10 loss/runtime/topology gate 通过；
- 9 节点 runtime baseline 与兼容 key 已冻结；
- independent-jobs 8+1 运行至少一次，或若队列暂不支持，有明确 Stage 3 前补跑 blocker；
- Checker 允许关闭。
