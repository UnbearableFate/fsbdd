from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from fsbdd.model_registry import (
    ExplicitMapping,
    MiscAssignment,
    RegistryError,
    build_logical_layer_registry,
    validate_parameter_ownership,
)


class TinyNeoX(nn.Module):
    def __init__(self, *, tied: bool = True) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="gpt_neox")
        self.gpt_neox = nn.Module()
        self.gpt_neox.embed_in = nn.Embedding(8, 4)
        self.gpt_neox.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.gpt_neox.final_layer_norm = nn.LayerNorm(4)
        self.embed_out = nn.Linear(4, 8, bias=False)
        if tied:
            self.embed_out.weight = self.gpt_neox.embed_in.weight


class TinyLlama(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="llama")
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.model.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8, bias=False)


class ThirdFamily(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="third")
        self.tokens = nn.Embedding(8, 4)
        self.stack = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.post = nn.LayerNorm(4)
        self.output = nn.Linear(4, 8, bias=False)


class ReorderedLlama(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="llama")
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.model = nn.Module()
        self.model.norm = nn.LayerNorm(4)
        self.model.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.model.embed_tokens = nn.Embedding(8, 4)


def unique_trainable(model: nn.Module) -> int:
    return len({id(parameter) for parameter in model.parameters() if parameter.requires_grad})


@pytest.mark.parametrize(("model", "family"), [(TinyNeoX(), "gpt_neox"), (TinyLlama(), "llama")])
def test_two_families_have_ordered_complete_coverage(model: nn.Module, family: str) -> None:
    registry = build_logical_layer_registry(model, sync_dtype_bytes=4)
    assert registry.family == family
    assert [layer.kind for layer in registry.layers] == ["embedding", "block", "block", "lm_head"]
    assert len(registry.parameters) == unique_trainable(model)
    assert registry.coverage.unique_trainable == registry.coverage.owned == len(registry.parameters)
    assert registry.coverage.duplicate_owners == 0
    assert registry.digest == registry.to_dict()["digest"]


def test_tied_owner_is_embedding_and_duplicate_fixture_rejects() -> None:
    model = TinyNeoX(tied=True)
    registry = build_logical_layer_registry(model)
    tied = [parameter for parameter in registry.parameters if len(parameter.aliases) > 1]
    assert len(tied) == 1
    assert tied[0].owner_layer == "embedding"
    assert set(tied[0].aliases) == {"gpt_neox.embed_in.weight", "embed_out.weight"}
    with pytest.raises(RegistryError, match="duplicate"):
        validate_parameter_ownership([("embedding", "same"), ("lm_head", "same")])


def test_misc_parameter_uses_smaller_adjacent_side_and_front_tie() -> None:
    registry = build_logical_layer_registry(TinyLlama(), sync_dtype_bytes=4)
    norm = next(parameter for parameter in registry.parameters if "model.norm.weight" in parameter.aliases)
    # Last block is smaller than the untied head in this fixture.
    assert norm.owner_layer == "block-1"


def test_unknown_structure_fails_closed_and_explicit_third_variation_passes() -> None:
    model = ThirdFamily()
    with pytest.raises(RegistryError, match="explicit"):
        build_logical_layer_registry(model)
    mapping = ExplicitMapping(
        family="third-explicit",
        embedding_path="tokens",
        block_paths=("stack.0", "stack.1"),
        head_path="output",
        misc=(MiscAssignment("post", left_layer=2, right_layer=3),),
    )
    registry = build_logical_layer_registry(model, explicit=mapping)
    assert registry.family == "third-explicit"
    assert len(registry.parameters) == unique_trainable(model)


def test_unlisted_auxiliary_parameter_is_rejected() -> None:
    model = TinyLlama()
    model.auxiliary = nn.Parameter(torch.ones(2))
    with pytest.raises(RegistryError, match="unowned"):
        build_logical_layer_registry(model)


def test_registration_order_does_not_change_registry_digest() -> None:
    assert build_logical_layer_registry(TinyLlama()).digest == build_logical_layer_registry(ReorderedLlama()).digest


def test_large_embedding_and_zero_parameter_head_are_supported_explicitly() -> None:
    model = ThirdFamily()
    model.tokens = nn.Embedding(4096, 4)
    model.post = nn.Identity()
    model.output = nn.Identity()
    mapping = ExplicitMapping(
        family="third-zero-head",
        embedding_path="tokens",
        block_paths=("stack.0", "stack.1"),
        head_path="output",
        misc=(),
    )
    registry = build_logical_layer_registry(model, explicit=mapping, sync_dtype_bytes=2)
    assert registry.layers[0].sync_bytes == 4096 * 4 * 2
    assert registry.layers[-1].parameter_count == 0
    assert registry.coverage.owned == unique_trainable(model)


def test_parameter_shared_across_blocks_is_rejected() -> None:
    model = TinyLlama()
    model.model.layers[1].weight = model.model.layers[0].weight
    with pytest.raises(RegistryError, match="shared"):
        build_logical_layer_registry(model)


def test_explicit_mapping_cannot_bypass_encoder_decoder_scope() -> None:
    model = ThirdFamily()
    model.config.is_encoder_decoder = True
    mapping = ExplicitMapping(
        family="forbidden-encoder-decoder",
        embedding_path="tokens",
        block_paths=("stack.0", "stack.1"),
        head_path="output",
        misc=(MiscAssignment("post", left_layer=2, right_layer=3),),
    )
    with pytest.raises(RegistryError, match="encoder-decoder"):
        build_logical_layer_registry(model, explicit=mapping)


def test_unclassified_buffer_fails_closed_for_explicit_mapping() -> None:
    model = ThirdFamily()
    model.register_buffer("training_running_state", torch.ones(2), persistent=True)
    mapping = ExplicitMapping(
        family="third-with-unknown-buffer",
        embedding_path="tokens",
        block_paths=("stack.0", "stack.1"),
        head_path="output",
        misc=(MiscAssignment("post", left_layer=2, right_layer=3),),
    )
    with pytest.raises(RegistryError, match="buffer"):
        build_logical_layer_registry(model, explicit=mapping)
