"""Gym-style hospital bed allocation environment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral
from typing import Literal, TypedDict

import numpy as np

Severity = Literal["low", "medium", "high"]
SEVERITY_LEVEL: dict[Severity, int] = {"low": 0, "medium": 1, "high": 2}


# Scenario configuration and the fixed rewards/penalties used by the environment.
@dataclass(frozen=True)
class Task:
    beds: int
    initial_patients: int
    max_steps: int = 5
    arrival_rate: float = 1.0


STANDARD_TASK = Task(beds=8, initial_patients=8, arrival_rate=1.5)

TREATMENT_REWARD: dict[Severity, float] = {"low": 1.0, "medium": 2.0, "high": 3.0}
EMERGENCY_BONUS = 1.0
HIGH_UNTREATED_PENALTY = -2.0
EMERGENCY_UNTREATED_PENALTY = -1.5
WASTED_BED_PENALTY = -0.3
DETERIORATION_PENALTY = -0.5
EMERGENCY_ARRIVAL_PROBABILITY = 0.2
DETERIORATION_PROBABILITY = 0.2


# These typed dictionaries describe the information shared between the
# environment and an agent, without exposing every internal patient detail.
class PatientObservation(TypedDict):
    id: int
    severity: Severity
    emergency: bool


class Observation(TypedDict):
    beds: int
    patients: list[PatientObservation]
    step: int
    max_steps: int
    done: bool


class StepInfo(TypedDict):
    allocated: int
    treated: int
    treated_patients: list[PatientObservation]
    arrivals: int
    emergency_arrivals: int
    occupied_beds: int
    reward_components: dict[str, float]


# Patient data and the hospital simulation environment.
@dataclass
class Patient:
    id: int
    severity: Severity
    emergency: bool = False
    remaining_stay: int = 0

    def observe(self) -> PatientObservation:
        # Return only the details needed to make a decision about the queue.
        return {
            "id": self.id,
            "severity": self.severity,
            "emergency": self.emergency,
        }


class HospitalEnvironment:
    """Episodic bed-allocation environment for an external RL agent."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed: int | None = None) -> Observation:
        # Start a new episode: clear progress and create the initial waiting queue.
        if seed is not None:
            self.seed = seed
        self.config = STANDARD_TASK
        self.rng = np.random.default_rng(self.seed)
        self.step_number = 0
        self.total_reward = 0.0
        self.next_patient_id = 1
        self.waiting: list[Patient] = []
        self.occupied: list[Patient] = []
        for _ in range(self.config.initial_patients):
            self.waiting.append(self._new_patient())
        self.history: list[dict[str, int | float]] = []
        self._record_history(
            reward=0.0,
            reward_components={},
            arrivals=0,
            emergency_arrivals=0,
        )
        return self.state()

    @property
    def done(self) -> bool:
        return self.step_number >= self.config.max_steps

    @property
    def beds_available(self) -> int:
        return self.config.beds - len(self.occupied)

    def state(self) -> Observation:
        # Present the current situation in a form the agent can inspect.
        return {
            "beds": self.beds_available,
            "patients": [patient.observe() for patient in self.waiting],
            "step": self.step_number,
            "max_steps": self.config.max_steps,
            "done": self.done,
        }

    def grade(self) -> float:
        """Map raw return monotonically into the API's (0.001, 0.999) range."""
        scale = max(float(self.config.max_steps), 1.0)
        normalized_return = float(np.clip(self.total_reward / scale, -60.0, 60.0))
        logistic = 1.0 / (1.0 + np.exp(-normalized_return))
        return float(0.001 + 0.998 * logistic)

    # Apply one allocation decision, advance the episode, and calculate reward.
    def step(self, allocate: int) -> tuple[Observation, float, bool, StepInfo]:
        if self.done:
            raise RuntimeError("Episode is complete; reset before taking another step.")
        if isinstance(allocate, bool) or not isinstance(allocate, Integral):
            raise ValueError("allocate must be an integer.")
        allocate = int(allocate)
        if allocate < 0:
            raise ValueError("allocate must be zero or greater.")
        if allocate > min(self.beds_available, len(self.waiting)):
            raise ValueError(
                "allocate cannot exceed available beds or the number of waiting patients."
            )

        # Beds freed at the end of the previous step are already available here.
        beds_available_at_step_start = self.beds_available
        reward_components = {
            "treated": 0.0,
            "emergency_bonus": 0.0,
            "untreated_high": 0.0,
            "untreated_emergency": 0.0,
            "wasted_bed": 0.0,
            "deterioration": 0.0,
        }

        # Admit the selected patients and award treatment/emergency rewards.
        treated = self._select_patients(allocate)
        for patient in treated:
            reward_components["treated"] += TREATMENT_REWARD[patient.severity]
            if patient.emergency:
                reward_components["emergency_bonus"] += EMERGENCY_BONUS
            patient.remaining_stay = int(self.rng.integers(1, 4))
            self.occupied.append(patient)

        # Penalize idle capacity and patients left waiting; waiting patients may worsen.
        if self.waiting:
            idle_beds = max(0, beds_available_at_step_start - len(treated))
            wasted = min(idle_beds, len(self.waiting))
            reward_components["wasted_bed"] = WASTED_BED_PENALTY * wasted

        still_waiting: list[Patient] = []
        for patient in self.waiting:
            if patient.severity == "high":
                reward_components["untreated_high"] += HIGH_UNTREATED_PENALTY
            if patient.emergency:
                reward_components["untreated_emergency"] += (
                    EMERGENCY_UNTREATED_PENALTY
                )
            if (
                patient.severity != "high"
                and self.rng.random() < DETERIORATION_PROBABILITY
            ):
                patient.severity = self._next_severity(patient.severity)
                reward_components["deterioration"] += DETERIORATION_PENALTY
            still_waiting.append(patient)
        self.waiting = still_waiting

        # Add new patients except after the final step, then return the transition.
        arrivals = (
            []
            if self.step_number + 1 >= self.config.max_steps
            else self._generate_arrivals()
        )
        self.waiting.extend(arrivals)
        self.step_number += 1
        # Advance treatment stays after this step so the next observation reflects
        # beds that have become free before the next action is selected.
        self._advance_discharges()

        reward = float(sum(reward_components.values()))
        self.total_reward += reward
        info: StepInfo = {
            "allocated": allocate,
            "treated": len(treated),
            "treated_patients": [patient.observe() for patient in treated],
            "arrivals": len(arrivals),
            "emergency_arrivals": sum(patient.emergency for patient in arrivals),
            "occupied_beds": len(self.occupied),
            "reward_components": reward_components,
        }
        self._record_history(
            reward=reward,
            reward_components=reward_components,
            arrivals=len(arrivals),
            emergency_arrivals=int(info["emergency_arrivals"]),
        )
        return self.state(), reward, self.done, info

    def _record_history(
        self,
        reward: float,
        reward_components: dict[str, float],
        arrivals: int,
        emergency_arrivals: int,
    ) -> None:
        # Save a per-step snapshot for analysis; this is separate from the Q-table.
        severity_counts = {
            severity: sum(patient.severity == severity for patient in self.waiting)
            for severity in ("high", "medium", "low")
        }
        self.history.append(
            {
                "step": self.step_number,
                "waiting": len(self.waiting),
                "waiting_high": severity_counts["high"],
                "waiting_medium": severity_counts["medium"],
                "waiting_low": severity_counts["low"],
                "waiting_emergency": sum(
                    patient.emergency for patient in self.waiting
                ),
                "occupied_beds": len(self.occupied),
                "available_beds": self.beds_available,
                "arrivals": arrivals,
                "emergency_arrivals": emergency_arrivals,
                "reward": reward,
                "cumulative_reward": self.total_reward,
                **{
                    f"reward_{name}": amount
                    for name, amount in reward_components.items()
                },
            }
        )

    def _new_patient(self, emergency: bool = False) -> Patient:
        # Regular patients get weighted severity; emergencies are always high severity.
        severity = self.rng.choice(
            ("low", "medium", "high"),
            p=(0.5, 0.3, 0.2),
        )
        if emergency:
            severity = "high"
        patient = Patient(
            id=self.next_patient_id,
            severity=severity,  # type: ignore[arg-type]
            emergency=emergency,
        )
        self.next_patient_id += 1
        return patient

    def _select_patients(self, allocate: int) -> list[Patient]:
        # Treat emergencies first, then higher severity, then earlier arrivals.
        priority = sorted(
            self.waiting,
            key=lambda patient: (
                patient.emergency,
                SEVERITY_LEVEL[patient.severity],
                -patient.id,
            ),
            reverse=True,
        )
        treated = priority[:allocate]
        treated_ids = {patient.id for patient in treated}
        self.waiting = [
            patient for patient in self.waiting if patient.id not in treated_ids
        ]
        return treated

    @staticmethod
    def _next_severity(severity: Severity) -> Severity:
        # Deterioration moves a patient up one severity level.
        return "medium" if severity == "low" else "high"

    def _advance_discharges(self) -> None:
        # Reduce existing treatment stays and free beds for patients who finish.
        remaining: list[Patient] = []
        for patient in self.occupied:
            patient.remaining_stay -= 1
            if patient.remaining_stay > 0:
                remaining.append(patient)
        self.occupied = remaining

    def _generate_arrivals(self) -> list[Patient]:
        # Generate Poisson-distributed regular arrivals plus a possible emergency.
        arrivals = [
            self._new_patient()
            for _ in range(int(self.rng.poisson(self.config.arrival_rate)))
        ]
        if self.rng.random() < EMERGENCY_ARRIVAL_PROBABILITY:
            arrivals.append(self._new_patient(emergency=True))
        return arrivals


# REINFORCEMENT LEARNING STARTS HERE.
# The agent learns Q-values: estimates of the long-term reward for each action
# (number of patients treated) in each summarized situation.
class QLearningAgent:
    """Tabular Q-learning agent for choosing how many waiting patients to treat."""

    def __init__(
        self,
        training_episodes: int = 600,
        learning_rate: float = 0.15,
        discount_factor: float = 0.95,
        seed: int = 2026,
    ) -> None:
        # Validate the learning settings before initializing the tables and metrics.
        if training_episodes < 1:
            raise ValueError("training_episodes must be positive.")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1].")
        if not 0 <= discount_factor < 1:
            raise ValueError("discount_factor must be in [0, 1).")
        self.training_episodes = training_episodes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.seed = seed
        self.q_table: defaultdict[tuple[int, ...], dict[int, float]] = defaultdict(
            dict
        )
        self.episode_returns: list[float] = []
        self.training_metrics: list[dict[str, float | int]] = []
        self.trained_q_table: dict[tuple[int, ...], dict[int, float]] = {}
        self.execution_metrics: list[dict[str, object]] = []
        self.trained = False

    # Convert the queue and episode progress into a compact Q-table state key.
    @staticmethod
    def state_key(observation: Observation) -> tuple[int, ...]:
        patients = observation["patients"]
        high = sum(patient["severity"] == "high" for patient in patients)
        medium = sum(patient["severity"] == "medium" for patient in patients)
        low = sum(patient["severity"] == "low" for patient in patients)
        emergencies = sum(patient["emergency"] for patient in patients)
        # Cap queue counts so unusually large queues do not create endless states.
        queue_cap = 2 * STANDARD_TASK.beds
        return (
            observation["beds"],
            min(high, queue_cap),
            min(medium, queue_cap),
            min(low, queue_cap),
            min(emergencies, queue_cap),
            observation["max_steps"] - observation["step"],
        )

    # The action is how many patients to treat: from zero up to available capacity.
    @staticmethod
    def valid_actions(observation: Observation) -> range:
        maximum = min(observation["beds"], len(observation["patients"]))
        return range(maximum + 1)

    def q_value(self, observation: Observation, action: int) -> float:
        # Unseen state/action pairs start with a Q-value of zero.
        return self.q_table.get(self.state_key(observation), {}).get(action, 0.0)

    # Choose the valid action with the highest Q-value.
    def choose_action(self, observation: Observation) -> int:
        actions = self.valid_actions(observation)
        state = self.state_key(observation)
        action_values = self.q_table.get(state, {})
        return max(actions, key=lambda action: (action_values.get(action, 0.0), action))

    # Update one Q-value after the environment supplies the reward and next state.
    def learn_from_transition(
        self,
        observation: Observation,
        action: int,
        reward: float,
        next_observation: Observation,
        reward_components: dict[str, float] | None = None,
    ) -> tuple[float, float]:
        """Apply one online Q-learning update and record the executed transition."""
        if action not in self.valid_actions(observation):
            raise ValueError("action must be valid for the supplied observation.")

        # Separate positive rewards from penalties for reporting the update clearly.
        components = reward_components or {}
        positive_reward = sum(max(amount, 0.0) for amount in components.values())
        penalty = -sum(min(amount, 0.0) for amount in components.values())
        if not components:
            positive_reward = max(reward, 0.0)
            penalty = max(-reward, 0.0)

        # Estimate the target from this reward plus the best known future value.
        state = self.state_key(observation)
        next_state = self.state_key(next_observation)
        current_q = self.q_table[state].get(action, 0.0)
        if next_observation["done"]:
            target = reward
        else:
            next_actions = self.valid_actions(next_observation)
            next_values = self.q_table.get(next_state, {})
            target = reward + self.discount_factor * max(
                next_values.get(next_action, 0.0)
                for next_action in next_actions
            )

        # Move the old estimate partway toward the target, using the learning rate.
        td_error = target - current_q
        updated_q = current_q + self.learning_rate * td_error
        self.q_table[state][action] = updated_q
        self.execution_metrics.append(
            {
                "step": observation["step"] + 1,
                "state": state,
                "action": action,
                "reward": reward,
                "gross_reward": positive_reward,
                "penalty": penalty,
                "final_result": reward,
                "reward_components": components.copy(),
                "q_before": current_q,
                "q_after": updated_q,
                "td_error": td_error,
            }
        )
        return updated_q, td_error

    def train(self) -> None:
        # Clear old results so calling train() again starts a fresh training run.
        self.q_table.clear()
        self.episode_returns.clear()
        self.training_metrics.clear()
        self.trained_q_table.clear()
        self.execution_metrics.clear()
        self.trained = False
        rng = np.random.default_rng(self.seed)
        alpha = self.learning_rate
        gamma = self.discount_factor
        total_transitions = 0

        # Each episode is one full run; epsilon gradually shifts choices
        # from random exploration toward the best-known action.
        for episode in range(self.training_episodes):
            environment = HospitalEnvironment(seed=self.seed + episode)
            observation = environment.state()
            episode_return = 0.0
            exploration_steps = 0
            episode_steps = 0
            progress = episode / max(self.training_episodes - 1, 1)
            epsilon = max(0.03, 0.35 * (1.0 - progress))

            while not observation["done"]:
                state = self.state_key(observation)
                actions = self.valid_actions(observation)
                if rng.random() < epsilon:
                    action = int(rng.choice(list(actions)))
                    exploration_steps += 1
                else:
                    action = self.choose_action(observation)

                # Take the action, observe its outcome, and update this Q-table entry.
                next_observation, reward, done, _ = environment.step(action)
                next_state = self.state_key(next_observation)
                current_q = self.q_table[state].get(action, 0.0)
                if done:
                    target = reward
                else:
                    next_actions = self.valid_actions(next_observation)
                    next_values = self.q_table.get(next_state, {})
                    target = reward + gamma * max(
                        next_values.get(next_action, 0.0)
                        for next_action in next_actions
                    )
                self.q_table[state][action] = current_q + alpha * (
                    target - current_q
                )
                episode_return += reward
                episode_steps += 1
                observation = next_observation

            # Keep episode-level results and learning-progress measurements.
            self.episode_returns.append(episode_return)
            total_transitions += episode_steps
            recent_returns = self.episode_returns[-100:]
            state_action_count = sum(len(values) for values in self.q_table.values())
            self.training_metrics.append(
                {
                    "episode": episode + 1,
                    "episode_return": episode_return,
                    "rolling_mean_return": float(np.mean(recent_returns)),
                    "rolling_std_return": (
                        float(np.std(recent_returns, ddof=1))
                        if len(recent_returns) > 1
                        else 0.0
                    ),
                    "epsilon": epsilon,
                    "exploration_rate": exploration_steps / episode_steps,
                    "exploration_decisions": exploration_steps,
                    "transitions": episode_steps,
                    "total_transitions": total_transitions,
                    "learned_states": len(self.q_table),
                    "state_action_values": state_action_count,
                    "mean_actions_per_state": (
                        state_action_count / len(self.q_table)
                        if self.q_table
                        else 0.0
                    ),
                }
            )

        # Preserve a snapshot of the completed training table.
        self.trained = True
        self.trained_q_table = {
            state: values.copy() for state, values in self.q_table.items()
        }
