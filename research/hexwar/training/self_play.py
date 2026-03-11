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
from ..agents.neural.ppo_agent import PPOAgent
from ..engine.game_engine import HexWarEnv
from ..engine.turn_resolver import TurnResult, resolve_player_turn
from ..engine.types import PLAYER_IDS, OrderMap
from ..engine.unit_generator import generate_units
from .reward import DEFAULT_REWARD_CONFIG, RewardConfig, compute_step_reward
from .rollout_buffer import RolloutBuffer, Transition


class SelfPlayCollector:
    """
    Collect PPO rollouts via self-play.

    Args:
        agent:          The learning agent (used for all 6 players unless
                        opponent_pool is provided).
        n_episodes:     Number of complete games to run per collect() call.
        max_turns:      Hard turn cap per game.
        reward_config:  Reward hyperparameters.
        opponent_pool:  Optional list of past-checkpoint agents to use as
                        opponents. If provided, 5 of the 6 player slots are
                        randomly filled from the pool.
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
        self.opponent_pool = opponent_pool or [agent]
        self.seed = seed

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
        """Run one game and append transitions to buffers."""
        import random as _random
        rng = _random.Random(self.seed + episode_idx)

        # Build agent assignment: learner plays one random slot
        learner_id = rng.choice(PLAYER_IDS)
        agents: dict[str, BaseAgent] = {}
        for pid in PLAYER_IDS:
            if pid == learner_id:
                agents[pid] = self.agent
            else:
                agents[pid] = rng.choice(self.opponent_pool)

        env = HexWarEnv(agents={}, seed=self.seed + episode_idx, max_turns=self.max_turns)

        # Manual step loop to capture per-step observations
        while env.phase != "end":
            if env.phase == "generateUnits":
                generate_units(env.board, env.stats)
                env.turn.turn_number += 1
                env.turn.active_ai_index = 0
                if env.turn.turn_number > self.max_turns:
                    env._end_game(winner_id=None)
                    break
                env.phase = "playerTurn"
                continue

            live = [p for p in env.players if not p.is_eliminated]
            idx = env.turn.active_ai_index
            if idx >= len(live):
                env.phase = "generateUnits"
                continue

            player = live[idx]
            agent = agents.get(player.id)

            board_before = deepcopy(env.board)
            players_before = deepcopy(env.players)

            orders: OrderMap = {}
            value_est = torch.tensor(0.0)
            chosen_edges = torch.zeros(0, dtype=torch.long)
            chosen_fracs = torch.zeros(0)
            log_p = torch.tensor(0.0)
            acting_mask_t = torch.zeros(0, dtype=torch.bool)

            if agent is not None and isinstance(agent, PPOAgent) and player.id == learner_id:
                # Collect with gradient-free forward
                with torch.no_grad():
                    from ..agents.neural.encoder import encode_board as _enc
                    data = _enc(env.board, player.id)
                    data = data.to(agent.device)

                    node_keys = data.node_keys
                    acting_mask_t = torch.tensor(
                        [
                            env.board[k].owner == player.id and env.board[k].units > 0
                            for k in node_keys
                        ],
                        dtype=torch.bool,
                        device=agent.device,
                    )

                    ml, alpha, beta, val = agent.model(
                        data.x, data.edge_index, data.edge_attr, data.u,
                        acting_mask_t, batch=None
                    )

                    src = data.edge_index[0]
                    valid_src = acting_mask_t[src]
                    ml = ml.masked_fill(~valid_src, -1e9)

                    import torch.nn.functional as F
                    from torch.distributions import Beta as _Beta

                    # Select best edge per source tile
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

                    # Build orders from selected edges
                    dst = data.edge_index[1]
                    for i, ei in enumerate(sel_edges.tolist()):
                        from_key = node_keys[src[ei].item()]
                        to_key = node_keys[dst[ei].item()]
                        ft = env.board.get(from_key)
                        if ft is None or ft.owner != player.id or ft.units == 0:
                            continue
                        frac_val = sel_fracs[i].item()
                        units = max(1, round(frac_val * ft.units))
                        from ..engine.types import MovementOrder
                        orders[from_key] = MovementOrder(
                            from_key=from_key, to_key=to_key, requested_units=units
                        )
            elif agent is not None:
                orders = agent(env.board, player.id, env.players, env.stats)

            result: TurnResult = resolve_player_turn(
                board=env.board,
                players=env.players,
                orders=orders,
                player_id=player.id,
                stats=env.stats,
            )

            env.board = result.board
            env.players = result.players
            env.stats = result.stats

            for p in env.players:
                if p.is_eliminated and p.id not in env._elimination_order:
                    env._elimination_order.append(p.id)

            reward = compute_step_reward(
                player_id=player.id,
                events=result.events,
                board_before=board_before,
                board_after=env.board,
                players_before=players_before,
                players_after=env.players,
                winner_id=result.winner_id,
                config=self.reward_config,
            )

            done = result.winner_id is not None

            # Only buffer transitions for the learner player
            if player.id == learner_id and isinstance(agent, PPOAgent):
                from ..agents.neural.encoder import encode_board as _enc2
                obs_data = _enc2(board_before, player.id)
                buffers[player.id].add(Transition(
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

            env.turn.active_ai_index += 1
