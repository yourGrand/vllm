# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.steering.steering_cache import SteeringCache
from vllm.steering.steering_metadata import SteeringMetadata

DEVICE = "cuda"


class TestSteeringCache:

    def test_allocate_and_store(self):
        """Allocate a slot, store a vector, and verify cache contents."""
        H = 128
        cache = SteeringCache(4, H, torch.float32, DEVICE)
        slot = cache.allocate()

        vec = torch.randn(H, dtype=torch.float32, device=DEVICE)
        cache.store(slot, vec)

        assert torch.equal(cache.cache[slot], vec)

    def test_allocate_all_slots(self):
        """Exhausting capacity raises RuntimeError on the next allocate."""
        max_slots = 3
        cache = SteeringCache(max_slots, 64, torch.float32, DEVICE)

        for _ in range(max_slots):
            cache.allocate()

        assert cache.num_free == 0
        with pytest.raises(RuntimeError, match="full"):
            cache.allocate()

    def test_release_and_reallocate(self):
        """A released slot can be reallocated and reused."""
        cache = SteeringCache(2, 64, torch.float32, DEVICE)
        s0 = cache.allocate()
        s1 = cache.allocate()
        assert cache.num_free == 0

        cache.release(s0)
        assert cache.num_free == 1

        s_new = cache.allocate()
        assert s_new == s0

        vec = torch.ones(64, dtype=torch.float32, device=DEVICE)
        cache.store(s_new, vec)
        assert torch.equal(cache.cache[s_new], vec)

    def test_release_invalid_slot(self):
        """Releasing an unallocated slot raises ValueError."""
        cache = SteeringCache(4, 64, torch.float32, DEVICE)
        with pytest.raises(ValueError, match="not currently allocated"):
            cache.release(0)

    def test_release_out_of_range(self):
        """Releasing a slot outside [0, max_slots) raises ValueError."""
        cache = SteeringCache(4, 64, torch.float32, DEVICE)
        with pytest.raises(ValueError, match="out of range"):
            cache.release(999)
        with pytest.raises(ValueError, match="out of range"):
            cache.release(-1)

    def test_store_to_unallocated_slot(self):
        """Storing to a slot that was never allocated raises ValueError."""
        cache = SteeringCache(4, 64, torch.float32, DEVICE)
        vec = torch.randn(64, dtype=torch.float32, device=DEVICE)
        with pytest.raises(ValueError, match="not currently allocated"):
            cache.store(0, vec)

    def test_store_shape_validation(self):
        """Storing a vector with the wrong shape raises ValueError."""
        cache = SteeringCache(4, 128, torch.float32, DEVICE)
        slot = cache.allocate()

        wrong_vec = torch.randn(64, dtype=torch.float32, device=DEVICE)
        with pytest.raises(ValueError, match="Expected vector of shape"):
            cache.store(slot, wrong_vec)


class TestSteeringMetadata:

    def test_update_metadata(self):
        """Mapped tokens receive correct values; unmapped stay at defaults."""
        meta = SteeringMetadata(16, DEVICE)
        num_tokens = 8
        token_to_slot = {0: 2, 3: 5, 7: 1}
        token_to_strength = {0: 1.0, 3: 0.5, 7: -0.3}

        meta.update(token_to_slot, token_to_strength, num_tokens)

        indices = meta.indices
        strengths = meta.strengths

        assert indices.shape == (num_tokens,)
        assert strengths.shape == (num_tokens,)

        assert indices[0].item() == 2
        assert indices[3].item() == 5
        assert indices[7].item() == 1

        unsteered = [1, 2, 4, 5, 6]
        for pos in unsteered:
            assert indices[pos].item() == -1
            assert abs(strengths[pos].item()) < 1e-6

        assert abs(strengths[0].item() - 1.0) < 1e-6
        assert abs(strengths[3].item() - 0.5) < 1e-6
        assert abs(strengths[7].item() - (-0.3)) < 1e-6

    def test_consecutive_updates(self):
        """A second update() call fully resets the previous batch."""
        meta = SteeringMetadata(16, DEVICE)

        meta.update({0: 3, 1: 4}, {0: 1.0, 1: 0.5}, 4)
        assert meta.indices[0].item() == 3
        assert meta.indices[1].item() == 4

        meta.update({2: 7}, {2: -1.0}, 6)

        assert meta.indices[0].item() == -1
        assert meta.indices[1].item() == -1
        assert meta.indices[2].item() == 7
        assert abs(meta.strengths[0].item()) < 1e-6
        assert abs(meta.strengths[1].item()) < 1e-6
        assert abs(meta.strengths[2].item() - (-1.0)) < 1e-6

    def test_reset(self):
        """Reset clears indices to -1 and strengths to 0."""
        meta = SteeringMetadata(8, DEVICE)
        meta.update({0: 1}, {0: 1.0}, 4)

        meta.reset()

        assert meta.indices.shape == (0,)
        assert meta.strengths.shape == (0,)
        assert (meta._indices == -1).all()
        assert (meta._strengths == 0.0).all()

    def test_num_tokens_exceeds_capacity(self):
        """Passing num_tokens beyond allocation raises ValueError."""
        meta = SteeringMetadata(8, DEVICE)
        with pytest.raises(ValueError, match="exceeds pre-allocated capacity"):
            meta.update({}, {}, 100)
