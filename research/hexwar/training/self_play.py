"""
Self-play rollout collection for PPO training.

Runs N_EPISODES games using the current policy for all 6 players
(with optional past-checkpoint opponents for diversity). Collects per-player
Transitions into RolloutBuffers.

Population-based self-play:
  - Maintain a pool of K checkpoint policies.
  - Each episode randomly assigns the "learner" to one player slot and
    fills the other 5 from the pool (or uses the current policy).
  - This prevents the learner from exploiting a fixed opponent.

Usage::

    collector = SelfPlayCollector(agent, n_episodes=32)
    buffers = collector.collect()
    # buffers is a dict[player_id, RolloutBuffer]
"""

from __future__ import annotations

from copy import deepcopy

import torch

from ..agents.base_agent import BaseAgent
from ..agents.neural.encoder import encode_board_with_history
from ..agents.neural.history_buffer import HistoryBuffer
from ..agents.neural.ppo_agent import PPOAgent
from ..engine.game_engine import HexWarEnv
from ..engine.turn_resolver import resolve_player_turn
from ..engine.types import PLAYER_IDS, MovementOrder, OrderMap
from ..engine.unit_generator import generate_units
from .reward import DEFAULT_REWARD_CONFIG, RewardConfig, compute_step_reward
from .rollout_buffer import RolloutBuffer, Transition


class SelfPlayCollector:
    """
    Collect PPO rollouts via self-play.

    Args:
        agent:          The learning agent (used for the learner slot).
        n_episodes:     Number of complete games to run per collect() call.
        max_turns:      Hard turn cap per game.
        reward_config:  Reward hyperparameters.
        opponent_pool:  List of past-checkpoint agents used as opponents.
                        Defaults to [agent] (current policy vs itself).
        seed:           Base RNG seed (incremented per episode).
    """

    def __init__(
        self,
        agent: PPOAgent,
        n_episodes: int = 16,
        max_turns: int = 300,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
        opponent_pool: list[BaseAgent] | None = None,
        seed: int = 0,
    ) -> None:
        self.agent = agent
        self.n_episodes = n_episodes
        self.max_turns = max_turns
        self.reward_config = reward_config
        self.opponent_pool: list[BaseAgent] = list(opponent_pool) if opponent_pool else [agent]
        self.seed = seed

    def add_to_pool(self, agent: BaseAgent) -> None:
        """Add a past-checkpoint agent to the opponent pool."""
        self.opponent_pool.append(agent)

    def collect(self) -> dict[str, RolloutBuffer]:
        """
        Run N_EPISODES games and return per-player RolloutBuffers.

        Returns:
            Dict mapping player_id → RolloutBuffer (filled, with returns
            already computed).
        """
        buffers: dict[str, RolloutBuffer] = {
            pid: RolloutBuffer() for pid in PLAYER_IDS
        }

        for ep in range(self.n_episodes):
            self._run_episode(ep, buffers)

        for buf in buffers.values():
            buf.compute_returns()

        return buffers

    def _run_episode(self, episode_idx: int, buffers: dict[str, RolloutBuffer]) -> None:
        """Run one game and append transitions to the learner's buffer."""
        import random as _random
        import torch.nn.functional as F
        from torch.distributions import Beta as _Beta

        rng = _random.Random(self.seed + episode_idx)

        # Randomly assign learner to one player slot; fill others from pool
        learner_id = rng.choice(PLAYER_IDS)
        agents: dict[str, BaseAgent] = {
            pid: (self.agent if pid == learner_id else rng.choice(self.opponent_pool))
            for pid in PLAYER_IDS
        }

        env = HexWarEnv(agents={}, seed=self.seed + episode_idx, max_turns=self.max_turns)

        # Reset history buffers on PPOAgent opponents (they maintain internal state)
        for pid, ag in agents.items():
            if isinstance(ag, PPOAgent) and pid != learner_id:
                ag.reset(env.board)

        # Per-player frame-stacking history buffers
        histories: dict[str, HistoryBuffer] = {
            pid: HistoryBuffer(k=self.agent.history_k) for pid in PLAYER_IDS
        }
        for hist in histories.values():
            hist.reset(env.board)

        while env.phase != "end":

            # ── Unit generation ───────────────────────────────────────
            if env.phase == "generateUnits":
                generate_units(env.board, env.stats)
                env.turn.turn_number += 1
                env.turn.active_ai_index = 0
                if env.turn.turn_number > self.max_turns:
                    env._end_game(winner_id=None)
                    break
                env.phase = "playerTurn"
                continue

            # ── Player turn ───────────────────────────────────────────
            # Mirror game_engine.py: iterate the fixed player_ids list so
            # mid-round eliminations never shift indices or skip players.
            idx = env.turn.active_ai_index
            if idx >= len(env.player_ids):
                env.phase = "generateUnits"
                continue

            pid = env.player_ids[idx]
            env.turn.active_ai_index += 1

            player = next((p for p in env.players if p.id == pid), None)
            if player is None or player.is_eliminated:
                continue

            agent = agents[pid]
            board_before = deepcopy(env.board)
            players_before = deepcopy(env.players)

            # Action defaults (for non-PPO or non-learner agents)
            orders: OrderMap = {}
            value_est = torch.tensor(0.0)
            chosen_edges = torch.zeros(0, dtype=torch.long)
            chosen_fracs = torch.zeros(0)
            log_p = torch.tensor(0.0)
            acting_mask_t = torch.zeros(0, dtype=torch.bool)
            obs_data = None

            if isinstance(agent, PPOAgent) and pid == learner_id:
                # Build frame-stacked observation (CPU) — keep on CPU for buffer
                frames = histories[pid].get_frames()
                frames[-1] = env.board  # ensure newest slot = current board

                obs_data = encode_board_with_history(frames, pid)
                data = obs_data.clone().to(agent.device)

                node_keys: list[str] = data.node_keys  # type: ignore[assignment]
                acting_mask_t = torch.tensor(
                    [env.board[k].owner == pid and env.board[k].units > 0
                     for k in node_keys],
                    dtype=torch.bool,
                    device=agent.device,
                )

                with torch.no_grad():
                    ml, alpha, beta, val = agent.model(
                        data.x, data.edge_index, data.edge_attr, data.u,
                        acting_mask_t, batch=None,
                    )

                src = data.edge_index[0]
                valid_src = acting_mask_t[src]
                ml = ml.masked_fill(~valid_src, -1e9)

                # Select best outgoing edge per source tile
                src_to_best: dict[int, int] = {}
                for ei in range(src.size(0)):
                    s = src[ei].item()
                    if not valid_src[ei]:
                        continue
                    if s not in src_to_best or ml[ei] > ml[src_to_best[s]]:
                        src_to_best[s] = ei

                sel_edges = torch.tensor(list(src_to_best.values()), dtype=torch.long)
                if sel_edges.numel() > 0:
                    frac_dist = _Beta(alpha[sel_edges], beta[sel_edges])
                    sel_fracs = frac_dist.sample()
                    lp_edge = F.log_softmax(ml, dim=0)[sel_edges]
                    lp_frac = frac_dist.log_prob(sel_fracs.clamp(1e-6, 1 - 1e-6))
                    log_p = (lp_edge + lp_frac).sum()
                else:
                    sel_fracs = torch.zeros(0)
                    log_p = torch.tensor(0.0)

                value_est = val
                chosen_edges = sel_edges
                chosen_fracs = sel_fracs

                # Convert selected edges → MovementOrders
                dst = data.edge_index[1]
                for i, ei in enumerate(sel_edges.tolist()):
                    from_key = node_keys[src[ei].item()]
                    to_key = node_keys[dst[ei].item()]
                    ft = env.board.get(from_key)
                    if ft is None or ft.owner != pid or ft.units == 0:
                        continue
                    units = max(1, round(sel_fracs[i].item() * ft.units))
                    orders[from_key] = MovementOrder(
                        from_key=from_key, to_key=to_key, requested_units=units,
                    )

            else:
                orders = agent(env.board, pid, env.players, env.stats)

            # Resolve turn
            result = resolve_player_turn(
                board=env.board,
                players=env.players,
                orders=orders,
                player_id=pid,
                stats=env.stats,
            )

            env.board = result.board
            env.players = result.players
            env.stats = result.stats

            for p in env.players:
                if p.is_eliminated and p.id not in env._elimination_order:
                    env._elimination_order.append(p.id)

            # Advance all history buffers with the post-move board state
            for hist in histories.values():
                hist.push(env.board)

            # Reward and transition storage (learner only)
            reward = compute_step_reward(
                player_id=pid,
                events=result.events,
                board_before=board_before,
                board_after=env.board,
                players_before=players_before,
                players_after=env.players,
                winner_id=result.winner_id,
                config=self.reward_config,
            )

            done = result.winner_id is not None

            if pid == learner_id and obs_data is not None:
                buffers[pid].add(Transition(
                    obs=obs_data,
                    acting_mask=acting_mask_t.cpu(),
                    chosen_edges=chosen_edges.cpu(),
                    chosen_fractions=chosen_fracs.cpu(),
                    log_prob=log_p.detach().cpu(),
                    value=value_est.detach().cpu().squeeze(),
                    reward=reward,
                    done=done,
                ))

            if done:
                env._end_game(winner_id=result.winner_id)
                break
