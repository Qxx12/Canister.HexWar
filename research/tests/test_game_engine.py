"""Integration tests for the headless game engine."""

from hexwar.agents.random_agent import RandomAgent
from hexwar.engine.game_engine import HexWarEnv
from hexwar.engine.types import PLAYER_IDS, GameResult


class TestHexWarEnv:
    def test_init_creates_board(self):
        env = HexWarEnv(seed=42)
        assert len(env.board) > 0
        assert len(env.players) == 6

    def test_start_tiles_populated(self):
        env = HexWarEnv(seed=42)
        start_tiles = [t for t in env.board.values() if t.is_start_tile]
        assert len(start_tiles) == len(PLAYER_IDS)

    def test_random_game_completes(self):
        """A game with all random agents must terminate."""
        agents = {pid: RandomAgent(seed=i) for i, pid in enumerate(PLAYER_IDS)}
        env = HexWarEnv(agents=agents, seed=42, max_turns=200)
        result = env.run()
        assert isinstance(result, GameResult)
        assert result.turns_played <= 200

    def test_random_game_has_winner_or_timeout(self):
        agents = {pid: RandomAgent(seed=i) for i, pid in enumerate(PLAYER_IDS)}
        env = HexWarEnv(agents=agents, seed=1, max_turns=100)
        result = env.run()
        # Either there's a winner (owns all start tiles) or no winner (timeout)
        if result.winner_id is not None:
            assert result.winner_id in PLAYER_IDS

    def test_reproducible_with_seed(self):
        """Same seed and agent seeds → identical results."""
        def make_env(seed):
            agents = {pid: RandomAgent(seed=i) for i, pid in enumerate(PLAYER_IDS)}
            return HexWarEnv(agents=agents, seed=seed, max_turns=50)

        r1 = make_env(7).run()
        r2 = make_env(7).run()
        assert r1.turns_played == r2.turns_played
        assert r1.winner_id == r2.winner_id

    def test_phase_transitions(self):
        """Step through phases manually and verify transitions."""
        env = HexWarEnv(seed=42, max_turns=5)
        assert env.phase == "playerTurn"

        # Run until first generateUnits phase
        steps = 0
        while env.phase == "playerTurn" and steps < 10:
            env.step()
            steps += 1

        # Should have transitioned to generateUnits or end
        assert env.phase in ("generateUnits", "end")

    def test_stats_populated_after_run(self):
        agents = {pid: RandomAgent(seed=i) for i, pid in enumerate(PLAYER_IDS)}
        env = HexWarEnv(agents=agents, seed=10, max_turns=50)
        result = env.run()
        assert len(result.stats) == len(PLAYER_IDS)

    def test_no_agents_game_ends(self):
        """A game with no agents (all players pass) ends at max_turns."""
        env = HexWarEnv(agents={}, seed=0, max_turns=5)
        result = env.run()
        # All players pass, so no conquest — should hit max_turns
        assert result.turns_played <= 5
