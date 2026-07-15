"""Deliberately wrong S1-06 learner loop used only for semantic RED."""

from __future__ import annotations

import inspect
import math

import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM


def forbidden_distributed_setup() -> None:
    """Wrong by construction: a learner must never create a process group."""

    torch.distributed.init_process_group("nccl")


class NaiveLearner:
    """Conflates microbatches and steps and never mutates model parameters."""

    def __init__(self, model: torch.nn.Module, batches: list[dict[str, torch.Tensor]]) -> None:
        self.model = model
        self.batches = batches
        self.global_step = 0
        self.shared_cursor = {"index": 0}
        self.events: list[dict[str, float | int]] = []

    def train(self, *, gradient_accumulation_steps: int) -> None:
        for batch in self.batches:
            self.model(**batch)
            # Wrong: a microbatch increments the only collapsed "global step".
            self.global_step += 1
            self.shared_cursor["index"] += 1
            # Wrong: no backward/optimizer/scheduler call, no safe boundary,
            # no loss or fragment progress, and a synthetic rank-rate sum.
            self.events.append(
                {
                    "global_step": self.global_step,
                    "throughput": float(batch["input_ids"].numel() * 8),
                }
            )


def _parameter_digest(model: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in model.parameters())


def main() -> int:
    torch.manual_seed(1606)
    model = GPTNeoXForCausalLM(
        GPTNeoXConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=32,
            use_cache=False,
        )
    )
    batches = []
    for index in range(10):
        input_ids = (torch.arange(32).reshape(2, 16) + index) % 64
        batches.append({"input_ids": input_ids, "labels": input_ids.clone()})

    before = _parameter_digest(model)
    learner = NaiveLearner(model, batches)
    learner.train(gradient_accumulation_steps=2)
    after = _parameter_digest(model)

    failures: list[str] = []
    if all(torch.equal(left, right) for left, right in zip(before, after, strict=True)):
        failures.append("LEARN-01/TEL-04: ten real HF forwards changed no parameter")
    if learner.global_step != 5:
        failures.append("PROG-04/PROG-05: microbatches advanced the collapsed global step")
    if any("loss" not in event for event in learner.events):
        failures.append("TEL-04: optimizer-boundary loss telemetry is missing")
    if any(not math.isfinite(float(event["throughput"])) for event in learner.events):
        failures.append("TEL-01: throughput is nonfinite")
    else:
        failures.append("TEL-01: throughput is a synthetic rank-rate sum without an interval")
    if any("fragment_local_steps" not in event for event in learner.events):
        failures.append("LEARN-01/PROG-04: per-fragment progress is absent")

    other = NaiveLearner(model, batches)
    other.shared_cursor = learner.shared_cursor
    if other.shared_cursor is learner.shared_cursor:
        failures.append("LEARN-01/LEARN-02: learners share a mutable data cursor")
    if "init_process_group" in inspect.getsource(forbidden_distributed_setup):
        failures.append("LEARN-02: learner source can initialize torch.distributed")

    if not failures:
        raise SystemExit("surrogate unexpectedly satisfied S1-06")
    for failure in failures:
        print(failure)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
