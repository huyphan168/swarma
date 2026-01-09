"""
Swarm Subnet 124 - PPO-Based Drone Flight Controller

This module implements an autonomous drone navigation agent trained using
Proximal Policy Optimization (PPO). The agent receives state observations
and outputs velocity commands to navigate from start to goal.

Training: python RL/ppo_drone_trainer.py --timesteps 100000
Testing:  python tests/test_rpc.py swarm/submission_template/ --seed 42
"""

import os
from pathlib import Path
from typing import Dict, Union, Optional

import numpy as np


class DroneFlightController:
    """
    PPO-trained autonomous drone flight controller.

    This controller uses a trained PPO policy to navigate drones from
    start positions to goal platforms. The policy processes state observations
    (position, velocity, orientation, goal direction) and outputs velocity commands.

    Observation Space:
        Dictionary with three keys:
        - "rgb": numpy array (96, 96, 4) - RGBA camera feed
        - "depth": numpy array (96, 96, 1) - Normalized depth map [0,1]
        - "state": numpy array (N,) - flight state vector containing:
            * Position (x, y, z) in meters
            * Orientation (roll, pitch, yaw)
            * Linear velocities (vx, vy, vz) in m/s
            * Angular velocities (roll_rate, pitch_rate, yaw_rate)
            * Action history (previous actions)
            * Altitude (normalized)
            * Search area vector (relative x, y, z) - ±10m accuracy

    Action Space:
        numpy array (5,) containing [vx, vy, vz, speed, yaw]
        - vx, vy, vz: velocity direction components, range [-1, 1]
        - speed: thrust multiplier, range [0, 1]
        - yaw: target yaw angle normalized, range [-1, 1] maps to [-π, π]
    """

    def __init__(self):
        """
        Initialize the flight controller by loading the trained PPO model.

        The model is loaded from ppo_policy.zip in the same directory.
        Falls back to a heuristic controller if no model is found.
        """
        self.model = None
        self.use_heuristic = True

        # Try to load trained PPO model
        model_path = Path(__file__).parent / "ppo_policy.zip"

        if model_path.exists():
            try:
                from stable_baselines3 import PPO
                self.model = PPO.load(str(model_path), device="cpu")
                self.use_heuristic = False
                print(f"[DroneFlightController] Loaded PPO model from {model_path}")
            except Exception as e:
                print(f"[DroneFlightController] Failed to load model: {e}")
                print("[DroneFlightController] Falling back to heuristic controller")
        else:
            print(f"[DroneFlightController] No model found at {model_path}")
            print("[DroneFlightController] Using heuristic controller")

        # Internal state for heuristic controller
        self._prev_action = np.zeros(5, dtype=np.float32)

    def act(self, observation: Union[Dict, np.ndarray]) -> np.ndarray:
        """
        Compute flight action for current observation.

        Args:
            observation: Either a dict with "rgb", "depth", "state" keys,
                        or directly the state array (N,)

        Returns:
            numpy array (5,) containing [vx, vy, vz, speed, yaw]
        """
        # Extract state vector from observation
        if isinstance(observation, dict):
            state = observation.get("state", np.zeros(16, dtype=np.float32))
        else:
            state = observation

        state = np.asarray(state, dtype=np.float32)

        if self.use_heuristic:
            action = self._heuristic_action(state)
        else:
            action = self._ppo_action(state)

        # Ensure action is valid
        action = np.asarray(action, dtype=np.float32).flatten()
        action = np.clip(action, -1.0, 1.0)
        action[3] = np.clip(action[3], 0.0, 1.0)  # Speed must be positive

        self._prev_action = action.copy()
        return action

    def _ppo_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get action from trained PPO model.

        Args:
            state: State vector from environment

        Returns:
            Action array (5,)
        """
        try:
            action, _ = self.model.predict(state, deterministic=True)
            return action.flatten()
        except Exception as e:
            print(f"[DroneFlightController] PPO prediction error: {e}")
            return self._heuristic_action(state)

    def _heuristic_action(self, state: np.ndarray) -> np.ndarray:
        """
        Heuristic goal-seeking controller.

        This provides a simple but effective baseline that:
        1. Moves toward the search area (approximate goal direction)
        2. Maintains stable flight with appropriate speed
        3. Adjusts yaw to face the goal

        Args:
            state: State vector containing position, velocity, etc.

        Returns:
            Action array (5,) containing [vx, vy, vz, speed, yaw]
        """
        # Parse state vector
        # Format: [pos(3), rpy(3), vel(3), ang_vel(3), action_history(...), altitude(1), search_vec(3)]

        # Position (0:3)
        pos = state[0:3] if len(state) >= 3 else np.zeros(3)

        # Orientation - roll, pitch, yaw (3:6)
        rpy = state[3:6] if len(state) >= 6 else np.zeros(3)

        # Velocity (6:9)
        vel = state[6:9] if len(state) >= 9 else np.zeros(3)

        # Search area vector is at the end of the state
        # State format: [pos(3), rpy(3), vel(3), ang_vel(3), actions(5*buffer), altitude(1), search_vec(3)]
        # With ACTION_BUFFER_SIZE=4, state_dim = 12 + 4*5 + 1 + 3 = 36
        search_vec = state[-3:] if len(state) >= 3 else np.zeros(3)

        # Direction to goal (search area gives approximate direction)
        goal_direction = search_vec.copy()
        goal_distance = np.linalg.norm(goal_direction)

        if goal_distance > 0.1:
            # Normalize direction
            goal_direction_norm = goal_direction / goal_distance

            # Proportional control toward goal
            # Scale velocity command based on distance
            if goal_distance > 5.0:
                # Far from goal: move at full speed
                speed = 0.9
                vel_scale = 1.0
            elif goal_distance > 1.0:
                # Getting closer: moderate speed
                speed = 0.6
                vel_scale = 0.8
            else:
                # Very close: slow approach
                speed = 0.3
                vel_scale = 0.5

            # Velocity command in direction of goal
            vx = goal_direction_norm[0] * vel_scale
            vy = goal_direction_norm[1] * vel_scale
            vz = goal_direction_norm[2] * vel_scale * 0.5  # More conservative vertical movement

            # Calculate desired yaw to face goal
            desired_yaw = np.arctan2(goal_direction[1], goal_direction[0])
            current_yaw = rpy[2] if len(rpy) >= 3 else 0.0

            # Yaw error (normalized to [-1, 1])
            yaw_error = desired_yaw - current_yaw
            # Wrap to [-pi, pi]
            while yaw_error > np.pi:
                yaw_error -= 2 * np.pi
            while yaw_error < -np.pi:
                yaw_error += 2 * np.pi
            # Normalize to [-1, 1]
            yaw_cmd = np.clip(yaw_error / np.pi, -1.0, 1.0)

        else:
            # At goal: hover
            vx, vy, vz = 0.0, 0.0, 0.0
            speed = 0.1
            yaw_cmd = 0.0

        # Smooth actions (blend with previous)
        alpha = 0.7  # Smoothing factor
        action = np.array([vx, vy, vz, speed, yaw_cmd], dtype=np.float32)
        action = alpha * action + (1 - alpha) * self._prev_action

        return action

    def reset(self):
        """
        Reset controller state at mission start.

        Called before each new mission. Resets internal state like
        action history and any accumulated errors.
        """
        self._prev_action = np.zeros(5, dtype=np.float32)
