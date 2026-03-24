"""Direct tests for RolloutBuffer and GAE computation."""

import pytest

try:
    import torch
    from hexwar.training.rollout_buffer import RolloutBuffer, Transition
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")


def _dummy_transition(reward: float = 0.0, done: bool = False, value: float = 0.0):
    """Build a minimal Transition with scalar tensors (no real PyG Data needed)."""
    return Transition(
        obs=None,                               # not used by compute_returns / iter_minibatches
        acting_mask=torch.tensor([True]),
        chosen_edges=torch.tensor([0]),
        chosen_fractions=torch.tensor([0.5]),
        log_prob=torch.tensor([-1.0]),
        value=torch.tensor(value),
        reward=reward,
        done=done,
    )


class TestRolloutBufferBasics:
    def test_empty_on_creation(self):
        buf = RolloutBuffer()
        assert len(buf) == 0

    def test_add_increases_len(self):
        buf = RolloutBuffer()
        buf.add(_dummy_transition())
        buf.add(_dummy_transition())
        assert len(buf) == 2

    def test_clear_empties_buffer(self):
        buf = RolloutBuffer()
        buf.add(_dummy_transition())
        buf.clear()
        assert len(buf) == 0

    def test_compute_returns_empty_is_noop(self):
        """compute_returns on an empty buffer must not raise."""
        buf = RolloutBuffer()
        buf.compute_returns()   # should not crash
        assert len(buf) == 0

    def test_custom_gamma_lam(self):
        buf = RolloutBuffer(gamma=0.9, lam=0.8)
        assert buf.gamma == 0.9
        assert buf.lam == 0.8


class TestComputeReturns:
    def test_advantage_and_return_set(self):
        """After compute_returns each transition has .advantage and .return_ attrs."""
        buf = RolloutBuffer(gamma=0.99, lam=0.95)
        for _ in range(4):
            buf.add(_dummy_transition(reward=1.0, value=0.5))
        buf.compute_returns()
        for tr in buf._transitions:
            assert hasattr(tr, "advantage")
            assert hasattr(tr, "return_")

    def test_terminal_transition_advantage(self):
        """
        Single terminal step: next_value = 0, done = True.
        GAE delta = reward + gamma * 0 * 0 - value = reward - value.
        advantage[0] = delta (gae = delta, no future).
        return_[0]   = advantage[0] + value.
        """
        buf = RolloutBuffer(gamma=0.99, lam=0.95)
        buf.add(_dummy_transition(reward=1.0, done=True, value=0.3))
        buf.compute_returns()
        tr = buf._transitions[0]
        expected_adv = 1.0 - 0.3   # delta = reward - value (done → next_value=0)
        assert abs(tr.advantage.item() - expected_adv) < 1e-5

    def test_non_terminal_propagation(self):
        """
        Two steps, no terminal. Verify backward propagation:
        Step 1 (t=0): reward=0, value=1.0  next_value=2.0
        Step 2 (t=1): reward=1, value=2.0  next_value=0 (end of buffer)

        delta[1] = 1 + 0.99*0 - 2.0 = -1.0  → gae[1] = -1.0
        delta[0] = 0 + 0.99*2.0 - 1.0 = 0.98 → gae[0] = 0.98 + 0.99*0.95*(-1.0)
        """
        buf = RolloutBuffer(gamma=0.99, lam=0.95)
        buf.add(_dummy_transition(reward=0.0, done=False, value=1.0))
        buf.add(_dummy_transition(reward=1.0, done=False, value=2.0))
        buf.compute_returns()

        adv1 = buf._transitions[1].advantage.item()
        expected_adv1 = 1.0 + 0.99 * 0.0 - 2.0   # = -1.0
        assert abs(adv1 - expected_adv1) < 1e-5

        adv0 = buf._transitions[0].advantage.item()
        delta0 = 0.0 + 0.99 * 2.0 - 1.0          # = 0.98
        expected_adv0 = delta0 + 0.99 * 0.95 * adv1
        assert abs(adv0 - expected_adv0) < 1e-5

    def test_return_equals_advantage_plus_value(self):
        """return_ == advantage + value for every transition."""
        buf = RolloutBuffer()
        for r in [0.5, -0.2, 1.0, 0.0]:
            buf.add(_dummy_transition(reward=r, value=0.4))
        buf.compute_returns()
        for tr in buf._transitions:
            expected = tr.advantage.item() + tr.value.item()
            assert abs(tr.return_.item() - expected) < 1e-5

    def test_done_cuts_future(self):
        """A done=True transition must not propagate future value backward."""
        buf = RolloutBuffer(gamma=0.99, lam=0.95)
        buf.add(_dummy_transition(reward=1.0, done=True,  value=0.0))
        buf.add(_dummy_transition(reward=5.0, done=False, value=0.0))
        buf.compute_returns()
        # With done=True at t=0, next_value from t=1 must be masked out.
        # Advantage at t=0 = reward - value = 1.0 - 0.0 = 1.0 (no future leaking)
        assert abs(buf._transitions[0].advantage.item() - 1.0) < 1e-5


class TestIterMinibatches:
    def test_all_transitions_yielded_exactly_once(self):
        buf = RolloutBuffer()
        n = 7
        for _ in range(n):
            buf.add(_dummy_transition())
        buf.compute_returns()

        seen = []
        for batch in buf.iter_minibatches(batch_size=3):
            seen.extend(batch)
        assert len(seen) == n

    def test_batch_size_respected(self):
        buf = RolloutBuffer()
        for _ in range(10):
            buf.add(_dummy_transition())
        buf.compute_returns()

        batches = list(buf.iter_minibatches(batch_size=4))
        # 10 transitions, batch_size=4 → 2 full batches + 1 remainder (2)
        assert len(batches) == 3
        assert len(batches[0]) == 4
        assert len(batches[1]) == 4
        assert len(batches[2]) == 2

    def test_single_element_batch(self):
        buf = RolloutBuffer()
        buf.add(_dummy_transition())
        buf.compute_returns()
        batches = list(buf.iter_minibatches(batch_size=1))
        assert len(batches) == 1
        assert len(batches[0]) == 1
