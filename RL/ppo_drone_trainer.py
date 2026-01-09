#!/usr/bin/env python3
"""
PPO Drone Controller Training Script

Implements a PPO-based drone controller for goal-reaching navigation tasks.
Uses dense reward shaping for efficient learning in the Swarm subnet environment.

Key features:
- Dense reward shaping (distance, velocity alignment, stability)
- Curriculum learning (start with simple straight-line flights)
- Focus on state-based policy (no vision for initial training)

Usage:
    python RL/ppo_drone_trainer.py --timesteps 100000 --seed 42
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor

# Add swarm to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.constants import SIM_DT, HORIZON_SEC, SPEED_LIMIT, GOAL_TOL


class DenseRewardWrapper(gym.Wrapper):
    """
    Wrapper that adds dense reward shaping for efficient PPO training.

    The original environment only gives sparse rewards based on mission success.
    This wrapper adds continuous feedback based on:
    - Distance to goal (primary objective)
    - Velocity alignment (encourage moving toward goal)
    - Stability penalties (discourage excessive angular velocity)
    - Progress bonus (reward getting closer)
    """

    def __init__(
        self,
        env: gym.Env,
        distance_weight: float = 1.0,
        velocity_weight: float = 0.5,
        stability_weight: float = 0.1,
        progress_weight: float = 2.0,
        success_bonus: float = 100.0,
        collision_penalty: float = -50.0,
    ):
        super().__init__(env)
        self.distance_weight = distance_weight
        self.velocity_weight = velocity_weight
        self.stability_weight = stability_weight
        self.progress_weight = progress_weight
        self.success_bonus = success_bonus
        self.collision_penalty = collision_penalty

        self._prev_distance = None
        self._initial_distance = None
        self._goal_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Get goal position from environment
        self._goal_pos = self.env.GOAL_POS.copy()

        # Initialize distance tracking
        drone_pos = self._get_drone_position(obs)
        self._initial_distance = np.linalg.norm(drone_pos - self._goal_pos)
        self._prev_distance = self._initial_distance

        return obs, info

    def _get_drone_position(self, obs: Dict) -> np.ndarray:
        """Extract drone position from observation state vector."""
        # State vector starts with position (x, y, z)
        state = obs.get("state", np.zeros(16))
        return state[0:3]

    def _get_drone_velocity(self, obs: Dict) -> np.ndarray:
        """Extract drone velocity from observation state vector."""
        # State: [pos(3), rpy(3), vel(3), ang_vel(3), ...]
        state = obs.get("state", np.zeros(16))
        return state[6:9]

    def _get_angular_velocity(self, obs: Dict) -> np.ndarray:
        """Extract angular velocity from observation state vector."""
        state = obs.get("state", np.zeros(16))
        return state[9:12]

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        # Calculate dense reward components
        drone_pos = self._get_drone_position(obs)
        drone_vel = self._get_drone_velocity(obs)
        ang_vel = self._get_angular_velocity(obs)

        current_distance = np.linalg.norm(drone_pos - self._goal_pos)

        # 1. Distance-based reward (exponential decay)
        # Normalized by initial distance for scale invariance
        normalized_dist = current_distance / (self._initial_distance + 1e-6)
        distance_reward = -self.distance_weight * normalized_dist

        # 2. Progress reward (delta distance)
        progress = self._prev_distance - current_distance
        progress_reward = self.progress_weight * progress

        # 3. Velocity alignment reward
        # Encourage velocity pointing toward goal
        goal_direction = self._goal_pos - drone_pos
        goal_direction_norm = np.linalg.norm(goal_direction)
        if goal_direction_norm > 0.1:
            goal_direction = goal_direction / goal_direction_norm
            vel_norm = np.linalg.norm(drone_vel)
            if vel_norm > 0.1:
                velocity_alignment = np.dot(drone_vel / vel_norm, goal_direction)
                # Scale by actual speed to encourage fast movement toward goal
                velocity_reward = self.velocity_weight * velocity_alignment * min(vel_norm / SPEED_LIMIT, 1.0)
            else:
                velocity_reward = 0.0
        else:
            velocity_reward = 0.0

        # 4. Stability penalty (discourage spinning)
        ang_vel_magnitude = np.linalg.norm(ang_vel)
        stability_penalty = -self.stability_weight * ang_vel_magnitude

        # 5. Success/failure bonuses
        success = info.get("success", False)
        collision = info.get("collision", False)

        terminal_reward = 0.0
        if success:
            # Bonus scaled by remaining time (faster = better)
            time_factor = 1.0 - (self.env._time_alive / HORIZON_SEC)
            terminal_reward = self.success_bonus * (0.5 + 0.5 * time_factor)
        elif collision:
            terminal_reward = self.collision_penalty

        # Combine all reward components
        shaped_reward = (
            distance_reward +
            progress_reward +
            velocity_reward +
            stability_penalty +
            terminal_reward
        )

        # Update tracking
        self._prev_distance = current_distance

        # Add reward breakdown to info for debugging
        info["reward_breakdown"] = {
            "distance": distance_reward,
            "progress": progress_reward,
            "velocity": velocity_reward,
            "stability": stability_penalty,
            "terminal": terminal_reward,
            "total": shaped_reward,
        }

        return obs, shaped_reward, terminated, truncated, info


class StateOnlyWrapper(gym.ObservationWrapper):
    """
    Wrapper that extracts only the state vector from observations.

    This simplifies training by focusing on the kinematic state
    rather than visual inputs (RGB, depth).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Get state dimension from original observation space
        original_state_space = env.observation_space["state"]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=original_state_space.shape,
            dtype=np.float32
        )

    def observation(self, obs: Dict) -> np.ndarray:
        """Extract state vector from dictionary observation."""
        return obs["state"].astype(np.float32)


class TrueGoalWrapper(gym.Wrapper):
    """
    Wrapper that replaces the noisy search vector with the true goal direction.

    This simplifies training by providing exact goal information.
    The last 3 elements of the state vector are replaced with the
    true direction to the goal platform.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._goal_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._goal_pos = self.env.GOAL_POS.copy()
        return self._modify_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._modify_obs(obs), reward, terminated, truncated, info

    def _modify_obs(self, obs: Dict) -> Dict:
        """Replace noisy search vector with true goal direction."""
        if self._goal_pos is None:
            return obs

        state = obs["state"].copy()
        drone_pos = state[0:3]

        # Replace last 3 elements with true goal vector
        true_goal_vec = (self._goal_pos - drone_pos).astype(np.float32)
        state[-3:] = true_goal_vec

        obs["state"] = state
        return obs


class TrainingCallback(BaseCallback):
    """Custom callback for logging training progress."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []

    def _on_step(self) -> bool:
        # Log episode statistics when episodes end
        if "episode" in self.locals.get("infos", [{}])[0]:
            info = self.locals["infos"][0]
            self.episode_rewards.append(info["episode"]["r"])
            self.episode_lengths.append(info["episode"]["l"])

        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-10:])
            mean_length = np.mean(self.episode_lengths[-10:])
            self.logger.record("rollout/mean_reward_10ep", mean_reward)
            self.logger.record("rollout/mean_length_10ep", mean_length)


def create_training_env(
    seed: int,
    challenge_type: int = 4,  # No obstacles for initial training
    use_dense_reward: bool = True,
    state_only: bool = True,
    use_true_goal: bool = True,  # Use true goal instead of noisy search area
) -> gym.Env:
    """
    Create a training environment with optional wrappers.

    Args:
        seed: Random seed for task generation
        challenge_type: 1-4, where 4 is no obstacles (easiest)
        use_dense_reward: Whether to apply dense reward shaping
        state_only: Whether to use only state vector observations
        use_true_goal: Whether to replace noisy search with true goal
    """
    # Generate task with specified challenge type
    task = random_task(
        sim_dt=SIM_DT,
        horizon=HORIZON_SEC,
        seed=seed,
    )
    # Override challenge type for curriculum learning
    task.challenge_type = challenge_type

    # Create base environment
    env = make_env(task, gui=False)

    # Apply wrappers in correct order
    # 1. True goal wrapper first (modifies observations)
    if use_true_goal:
        env = TrueGoalWrapper(env)

    # 2. Dense reward wrapper (needs goal position)
    if use_dense_reward:
        env = DenseRewardWrapper(env)

    # 3. State-only wrapper last (extracts state from dict)
    if state_only:
        env = StateOnlyWrapper(env)

    # Wrap with Monitor for logging
    env = Monitor(env)

    return env


def make_vec_env(
    n_envs: int,
    seed: int,
    challenge_type: int = 4,
    use_dense_reward: bool = True,
    state_only: bool = True,
    use_true_goal: bool = True,
) -> DummyVecEnv:
    """Create vectorized training environments."""

    def make_env_fn(env_seed):
        def _init():
            return create_training_env(
                seed=env_seed,
                challenge_type=challenge_type,
                use_dense_reward=use_dense_reward,
                state_only=state_only,
                use_true_goal=use_true_goal,
            )
        return _init

    env_fns = [make_env_fn(seed + i) for i in range(n_envs)]
    return DummyVecEnv(env_fns)


def train_ppo(
    total_timesteps: int = 100_000,
    seed: int = 42,
    n_envs: int = 4,
    challenge_type: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_steps: int = 2048,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    save_path: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    verbose: int = 1,
) -> PPO:
    """
    Train PPO agent for drone navigation.

    Args:
        total_timesteps: Total training timesteps
        seed: Random seed
        n_envs: Number of parallel environments
        challenge_type: Task difficulty (4=no obstacles, 1=hardest)
        learning_rate: PPO learning rate
        batch_size: Minibatch size
        n_steps: Steps per rollout
        n_epochs: PPO epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        save_path: Where to save the trained model
        log_dir: Tensorboard log directory
        verbose: Verbosity level

    Returns:
        Trained PPO model
    """
    # Create vectorized environment
    print(f"Creating {n_envs} parallel training environments...")
    print(f"Challenge type: {challenge_type} (4=no obstacles, 1=hardest)")

    vec_env = make_vec_env(
        n_envs=n_envs,
        seed=seed,
        challenge_type=challenge_type,
        use_dense_reward=True,
        state_only=True,
    )

    # Define policy network architecture
    # Using MLP for state-based observations
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],  # Policy network
            vf=[256, 256],  # Value network
        ),
    )

    # Create PPO model
    print("Initializing PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=seed,
        tensorboard_log=str(log_dir) if log_dir else None,
    )

    # Setup callbacks
    callbacks = [TrainingCallback()]

    # Checkpoint callback
    if save_path:
        checkpoint_dir = save_path.parent / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_drone",
        )
        callbacks.append(checkpoint_callback)

    # Train
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("=" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    print("=" * 60)
    print("Training complete!")

    # Save final model
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        print(f"\nModel saved to: {save_path}")

    vec_env.close()
    return model


def evaluate_model(
    model_path: Path,
    n_episodes: int = 10,
    seed: int = 42,
    gui: bool = False,
    use_true_goal: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a trained model.

    Args:
        model_path: Path to saved model
        n_episodes: Number of evaluation episodes
        seed: Random seed
        gui: Whether to show visualization
        use_true_goal: Whether to use true goal (should match training)

    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\nEvaluating model: {model_path}")
    print(f"Running {n_episodes} episodes...")
    print(f"Using true goal: {use_true_goal}")

    model = PPO.load(str(model_path))

    successes = []
    times = []
    scores = []

    for ep in range(n_episodes):
        task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=seed + ep)
        task.challenge_type = 4  # No obstacles for evaluation

        env = make_env(task, gui=gui)

        # Apply same wrappers as training
        if use_true_goal:
            env = TrueGoalWrapper(env)
        env = StateOnlyWrapper(env)

        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        successes.append(info.get("success", False))
        # Access time from the base environment
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        times.append(base_env._time_alive)
        scores.append(info.get("score", 0.0))

        env.close()

        status = "SUCCESS" if info.get("success") else "FAILED"
        print(f"  Episode {ep+1}: {status}, time={times[-1]:.2f}s, score={scores[-1]:.3f}")

    results = {
        "success_rate": np.mean(successes),
        "mean_time": np.mean(times),
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
    }

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Success rate: {results['success_rate']*100:.1f}%")
    print(f"Mean time: {results['mean_time']:.2f}s")
    print(f"Mean score: {results['mean_score']:.3f} (+/- {results['std_score']:.3f})")
    print("=" * 40)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO drone controller for Swarm subnet"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Total training timesteps (default: 100000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--n-envs", type=int, default=4,
        help="Number of parallel environments (default: 4)"
    )
    parser.add_argument(
        "--challenge-type", type=int, default=4, choices=[1, 2, 3, 4],
        help="Challenge type: 4=no obstacles, 1=hardest (default: 4)"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run evaluation after training"
    )
    parser.add_argument(
        "--eval-only", type=Path, default=None,
        help="Only evaluate existing model (skip training)"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Show GUI during evaluation"
    )
    args = parser.parse_args()

    # Paths
    output_dir = Path(__file__).parent.parent / "swarm" / "submission_template"
    model_path = output_dir / "ppo_policy.zip"
    log_dir = Path(__file__).parent / "logs"

    if args.eval_only:
        # Evaluation only mode
        evaluate_model(
            model_path=args.eval_only,
            n_episodes=10,
            seed=args.seed,
            gui=args.gui,
        )
    else:
        # Training mode
        model = train_ppo(
            total_timesteps=args.timesteps,
            seed=args.seed,
            n_envs=args.n_envs,
            challenge_type=args.challenge_type,
            learning_rate=args.lr,
            save_path=model_path,
            log_dir=log_dir,
            verbose=1,
        )

        if args.eval:
            evaluate_model(
                model_path=model_path,
                n_episodes=10,
                seed=args.seed,
                gui=args.gui,
            )

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Test locally:")
    print("   python tests/test_rpc.py swarm/submission_template/ --seed 42")
    print("\n2. Create submission zip:")
    print("   python tests/test_rpc.py swarm/submission_template/ --zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
