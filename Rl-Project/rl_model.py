"""Bed-allocation Markov decision process and stochastic hospital simulation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TypeAlias

import numpy as np
from scipy.stats import poisson


State: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True)
class SimulationConfig:
    total_beds: int = 10
    normal_beds: int = 5
    normal_arrivals: float = 2.0
    covid_arrivals: float = 2.0
    normal_discharges: float = 1.0
    covid_discharges: float = 1.0
    discount_rate: float = 0.8
    simulation_days: int = 30
    seed: int = 7

    @property
    def covid_beds(self) -> int:
        return self.total_beds - self.normal_beds

    def validate(self) -> None:
        if not 2 <= self.total_beds <= 15:
            raise ValueError("Total beds must be between 2 and 15.")
        if not 0 <= self.normal_beds <= self.total_beds:
            raise ValueError("Normal beds must be between zero and total beds.")
        rates = (
            self.normal_arrivals,
            self.covid_arrivals,
            self.normal_discharges,
            self.covid_discharges,
        )
        if any(rate < 0 or rate > 10 for rate in rates):
            raise ValueError("Arrival and discharge rates must be between 0 and 10.")
        if not 0 < self.discount_rate < 1:
            raise ValueError("Discount rate must be greater than zero and less than one.")
        if not 1 <= self.simulation_days <= 100:
            raise ValueError("Simulation days must be between 1 and 100.")


def _poisson_support(rate: float, limit: int) -> list[tuple[int, float, float]]:
    """Return count, probability, and conditional mean, folding the tail at limit."""
    support: list[tuple[int, float, float]] = []
    for count in range(limit):
        probability = float(poisson.pmf(count, rate))
        if probability > 1e-12:
            support.append((count, probability, float(count)))

    tail_probability = float(poisson.sf(limit - 1, rate))
    if tail_probability > 1e-12:
        tail_mean = rate * float(poisson.sf(limit - 2, rate)) / tail_probability
        support.append((limit, tail_probability, tail_mean))
    return support


class PolicySolver:
    """Value-iterate a finite state MDP over ward capacity and free beds."""

    def __init__(self, config: SimulationConfig) -> None:
        config.validate()
        self.config = config
        self.states: list[State] = []
        self._state_index: dict[State, int] = {}
        self._ward_rewards: list[list[np.ndarray]] = []
        self._ward_transitions: list[list[np.ndarray]] = []
        self.policy: dict[State, int] = {}
        self.values: dict[State, float] = {}
        self.iterations = 0
        self._create_states()
        self._create_transitions()
        self._solve()

    def _create_states(self) -> None:
        total = self.config.total_beds
        for normal_capacity in range(total + 1):
            covid_capacity = total - normal_capacity
            for normal_available in range(normal_capacity + 1):
                for covid_available in range(covid_capacity + 1):
                    state = (normal_capacity, normal_available, covid_available)
                    self._state_index[state] = len(self.states)
                    self.states.append(state)

    def _create_transitions(self) -> None:
        total = self.config.total_beds
        ward_parameters = (
            (
                self.config.normal_arrivals,
                self.config.normal_discharges,
                10.0,
            ),
            (
                self.config.covid_arrivals,
                self.config.covid_discharges,
                20.0,
            ),
        )
        for arrivals, discharges, unmet_cost in ward_parameters:
            request_support = _poisson_support(arrivals, total + 1)
            discharge_support = _poisson_support(discharges, total + 1)
            ward_rewards: list[np.ndarray] = []
            ward_transitions: list[np.ndarray] = []
            for capacity in range(total + 1):
                rewards = np.zeros(capacity + 1, dtype=float)
                transitions = np.zeros((capacity + 1, capacity + 1), dtype=float)
                for available in range(capacity + 1):
                    for request, request_probability, mean_request in request_support:
                        admitted = min(request, available)
                        unmet = max(mean_request - admitted, 0.0)
                        occupied_after_request = capacity - available + admitted
                        available_after_request = available - admitted
                        for returned, return_probability, _ in discharge_support:
                            discharged = min(returned, occupied_after_request)
                            probability = request_probability * return_probability
                            next_available = available_after_request + discharged
                            transitions[available, next_available] += probability
                            rewards[available] -= probability * unmet_cost * unmet
                ward_rewards.append(rewards)
                ward_transitions.append(transitions)
            self._ward_rewards.append(ward_rewards)
            self._ward_transitions.append(ward_transitions)

    @lru_cache(maxsize=None)
    def actions(self, state: State) -> tuple[tuple[int, State], ...]:
        normal_capacity, normal_available, covid_available = state
        min_action = -covid_available
        max_action = normal_available
        actions = []
        for action in range(min_action, max_action + 1):
            next_state = (
                normal_capacity - action,
                normal_available - action,
                covid_available + action,
            )
            actions.append((action, next_state))
        return tuple(actions)

    def _solve(self) -> None:
        total = self.config.total_beds
        value = np.zeros((total + 1, total + 1, total + 1), dtype=float)
        policy = np.zeros((total + 1, total + 1, total + 1), dtype=int)
        gamma = self.config.discount_rate

        for iteration in range(1, 101):
            updated = np.empty_like(value)
            updated.fill(0.0)
            future_by_capacity: list[np.ndarray] = []
            for normal_capacity in range(total + 1):
                covid_capacity = total - normal_capacity
                normal_transitions = self._ward_transitions[0][normal_capacity]
                covid_transitions = self._ward_transitions[1][covid_capacity]
                future_by_capacity.append(
                    normal_transitions
                    @ value[
                        normal_capacity,
                        : normal_capacity + 1,
                        : covid_capacity + 1,
                    ]
                    @ covid_transitions.T
                )

            for normal_capacity in range(total + 1):
                covid_capacity = total - normal_capacity
                for normal_available in range(normal_capacity + 1):
                    for covid_available in range(covid_capacity + 1):
                        state = (
                            normal_capacity,
                            normal_available,
                            covid_available,
                        )
                        best_value = -float("inf")
                        best_action = 0
                        for action, next_state in self.actions(state):
                            next_normal_capacity = next_state[0]
                            next_covid_capacity = total - next_normal_capacity
                            immediate_reward = (
                                self._ward_rewards[0][next_normal_capacity][
                                    next_state[1]
                                ]
                                + self._ward_rewards[1][next_covid_capacity][
                                    next_state[2]
                                ]
                                - 5.0 * abs(action)
                            )
                            action_value = immediate_reward + gamma * (
                                future_by_capacity[next_normal_capacity][
                                    next_state[1], next_state[2]
                                ]
                            )
                            if action_value > best_value + 1e-10 or (
                                abs(action_value - best_value) <= 1e-10
                                and abs(action) < abs(best_action)
                            ):
                                best_value = action_value
                                best_action = action
                        updated[normal_capacity, normal_available, covid_available] = (
                            best_value
                        )
                        policy[
                            normal_capacity, normal_available, covid_available
                        ] = best_action

            delta = float(np.max(np.abs(updated - value)))
            value = updated
            if delta < 1e-4:
                self.iterations = iteration
                break
        else:
            self.iterations = 100

        self.values = {
            state: float(value[state[0], state[1], state[2]])
            for state in self.states
        }
        self.policy = {
            state: int(policy[state[0], state[1], state[2]])
            for state in self.states
        }

    def action_for(self, state: State) -> int:
        return self.policy[state]

    def expected_reward(self, state: State) -> float:
        normal_capacity, normal_available, covid_available = state
        covid_capacity = self.config.total_beds - normal_capacity
        return float(
            self._ward_rewards[0][normal_capacity][normal_available]
            + self._ward_rewards[1][covid_capacity][covid_available]
        )


class HospitalSimulator:
    """Reproducible day-by-day simulator using the solver's policy."""

    def __init__(self, config: SimulationConfig) -> None:
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.day = 0
        self.normal_capacity = config.normal_beds
        self.covid_capacity = config.covid_beds
        self.normal_available = self.normal_capacity
        self.covid_available = self.covid_capacity
        self.total_unmet_normal = 0
        self.total_unmet_covid = 0
        self.total_moved = 0
        self.total_reward = 0.0
        self.history: list[dict[str, int | float]] = [
            {
                "day": 0,
                "normal_available": self.normal_available,
                "covid_available": self.covid_available,
                "normal_occupied": 0,
                "covid_occupied": 0,
                "normal_unmet": 0,
                "covid_unmet": 0,
                "reward": 0.0,
                "beds_moved": 0,
            }
        ]

    @property
    def state(self) -> State:
        return (
            self.normal_capacity,
            self.normal_available,
            self.covid_available,
        )

    @property
    def finished(self) -> bool:
        return self.day >= self.config.simulation_days

    def step(self, policy: PolicySolver) -> dict[str, int | float]:
        if self.finished:
            raise RuntimeError("The simulation has already reached its configured duration.")

        old_state = self.state
        action = policy.action_for(old_state)
        self.normal_capacity -= action
        self.covid_capacity += action
        self.normal_available -= action
        self.covid_available += action

        requests_normal = int(self.rng.poisson(self.config.normal_arrivals))
        requests_covid = int(self.rng.poisson(self.config.covid_arrivals))
        admissions_normal = min(requests_normal, self.normal_available)
        admissions_covid = min(requests_covid, self.covid_available)
        unmet_normal = requests_normal - admissions_normal
        unmet_covid = requests_covid - admissions_covid
        self.normal_available -= admissions_normal
        self.covid_available -= admissions_covid

        occupied_normal = self.normal_capacity - self.normal_available
        occupied_covid = self.covid_capacity - self.covid_available
        discharged_normal = min(
            int(self.rng.poisson(self.config.normal_discharges)), occupied_normal
        )
        discharged_covid = min(
            int(self.rng.poisson(self.config.covid_discharges)), occupied_covid
        )
        self.normal_available += discharged_normal
        self.covid_available += discharged_covid

        moved = abs(action)
        reward = -10 * unmet_normal - 20 * unmet_covid - 5 * moved
        self.day += 1
        self.total_unmet_normal += unmet_normal
        self.total_unmet_covid += unmet_covid
        self.total_moved += moved
        self.total_reward += reward
        record: dict[str, int | float] = {
            "day": self.day,
            "normal_available": self.normal_available,
            "covid_available": self.covid_available,
            "normal_occupied": self.normal_capacity - self.normal_available,
            "covid_occupied": self.covid_capacity - self.covid_available,
            "normal_unmet": unmet_normal,
            "covid_unmet": unmet_covid,
            "reward": float(reward),
            "beds_moved": moved,
            "action": action,
            "requests_normal": requests_normal,
            "requests_covid": requests_covid,
            "admissions_normal": admissions_normal,
            "admissions_covid": admissions_covid,
            "discharged_normal": discharged_normal,
            "discharged_covid": discharged_covid,
        }
        self.history.append(record)
        return record


if __name__ == "__main__":
    default_config = SimulationConfig()
    solver = PolicySolver(default_config)
    starting_state = (
        default_config.normal_beds,
        default_config.normal_beds,
        default_config.covid_beds,
    )
    print(f"Policy converged in {solver.iterations} iterations.")
    print(f"Recommended transfer: {solver.action_for(starting_state):+d} beds.")
