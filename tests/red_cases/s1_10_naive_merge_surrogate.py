"""Semantic RED for the intentionally wrong S1-10 cache-all merge surrogate."""

from __future__ import annotations

import unittest


def naive_current_relative_merge(current, declared_bases, locals_, weights):
    del declared_bases
    return [
        sum(
            weight * (current[index] - local[index])
            for weight, local in zip(weights, locals_, strict=True)
        )
        for index in range(len(current))
    ]


class CacheAllSource:
    def __init__(self, payloads):
        self.payloads = payloads
        self.active = 0
        self.maximum_active = 0

    def open_all(self):
        opened = []
        for payload in self.payloads:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            opened.append(payload)
        return opened


class WholeModelStore:
    def __init__(self):
        self.full_model_reads = 0

    def load_model(self):
        self.full_model_reads += 1
        return [b"fragment-0", b"fragment-1", b"fragment-2", b"fragment-3"]


def naive_outer_update(store, fragment_index):
    return store.load_model()[fragment_index]


def naive_old_buffer_nesterov(
    parameters, gradient, old_buffer, learning_rate, momentum
):
    next_buffer = [
        momentum * old + grad for old, grad in zip(old_buffer, gradient, strict=True)
    ]
    # Wrong: Nesterov must use the updated buffer.
    next_parameters = [
        value - learning_rate * (grad + momentum * old)
        for value, grad, old in zip(parameters, gradient, old_buffer, strict=True)
    ]
    return next_parameters, next_buffer


class S110SemanticRed(unittest.TestCase):
    def test_opt_01__declared_base_not_current_base(self):
        current = [12.0, -2.0]
        actual = naive_current_relative_merge(
            current,
            [[3.0, 5.0], current],
            [[1.0, 1.0], [8.0, -4.0]],
            [0.5, 0.5],
        )
        self.assertEqual(
            actual, [3.0, 3.0], "OPT-01 requires declared-base displacements"
        )

    def test_sync_06__only_one_local_payload_may_be_live(self):
        source = CacheAllSource([bytes(32) for _ in range(8)])
        source.open_all()
        self.assertLessEqual(
            source.maximum_active, 1, "SYNC-06 forbids cache-all M payloads"
        )

    def test_sync_07__steady_update_must_not_read_full_model(self):
        store = WholeModelStore()
        naive_outer_update(store, 2)
        self.assertEqual(store.full_model_reads, 0, "SYNC-07 forbids full-model reads")

    def test_opt_04__nesterov_uses_updated_buffer(self):
        parameters, buffer = naive_old_buffer_nesterov(
            [9.62, -4.24], [-1.0, 3.0], [2.0, -4.0], 0.1, 0.9
        )
        self.assertAlmostEqual(buffer[0], 0.8, delta=1e-6)
        self.assertAlmostEqual(buffer[1], -0.6, delta=1e-6)
        self.assertAlmostEqual(parameters[0], 9.648, delta=1e-6)
        self.assertAlmostEqual(parameters[1], -4.486, delta=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
