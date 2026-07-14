"""Deliberately wrong string registry; all S1-01 RED cases execute."""

from __future__ import annotations


names = [
    "gpt_neox.embed_in.weight",
    "gpt_neox.layers.0.attention.weight",
    "gpt_neox.final_layer_norm.weight",
    "embed_out.weight",
]


def naive_owner(name: str) -> str | None:
    parts = name.split(".")
    if parts[0] == "layers" and parts[1].isdigit():
        return f"block-{parts[1]}"
    if "embed" in name:
        return "embedding"
    if "head" in name:
        return "lm_head"
    return None


failures: list[str] = []
owners = [naive_owner(name) for name in names]
if owners[1] is None:
    failures.append("MODEL-01/02: root-only layers parser missed GPT-NeoX block")
if owners[2] is None:
    failures.append("MODEL-03/05: final norm remained unowned")
if owners[0] == owners[3] == "embedding":
    failures.append("MODEL-04: tied aliases would be emitted twice under one guessed bucket")
if naive_owner("unknown.layers.0.weight") is None:
    failures.append("MODEL-06: parser cannot distinguish ambiguity from unsupported structure")

if failures:
    raise AssertionError("\n".join(failures))
