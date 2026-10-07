import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from api import app
from rl_model import (
    EMERGENCY_ARRIVAL_PROBABILITY,
    HospitalEnvironment,
    Patient,
    QLearningAgent,
    STANDARD_TASK,
    Task,
)


class HospitalEnvironmentTests(unittest.TestCase):
    def test_fixed_environment_matches_project_specification(self) -> None:
        self.assertEqual(
            (STANDARD_TASK.beds, STANDARD_TASK.initial_patients),
            (8, 8),
        )
        self.assertEqual(
            STANDARD_TASK.arrival_rate,
            1.5,
        )

    def test_reset_returns_observation_and_is_repeatable_with_seed(self) -> None:
        first = HospitalEnvironment(seed=42)
        second = HospitalEnvironment(seed=42)

        self.assertEqual(first.state(), second.state())
        self.assertEqual(first.state()["beds"], 8)
        self.assertEqual(len(first.state()["patients"]), 8)
        self.assertNotIn("task", first.state())

    def test_episode_history_records_graph_series(self) -> None:
        environment = HospitalEnvironment(seed=42)
        self.assertEqual(len(environment.history), 1)
        self.assertEqual(environment.history[0]["step"], 0)

        environment.step(2)

        self.assertEqual(len(environment.history), 2)
        self.assertEqual(environment.history[-1]["step"], 1)
        self.assertEqual(
            environment.history[-1]["waiting"],
            environment.history[-1]["waiting_high"]
            + environment.history[-1]["waiting_medium"]
            + environment.history[-1]["waiting_low"],
        )
        self.assertIn("occupied_beds", environment.history[-1])
        self.assertIn("cumulative_reward", environment.history[-1])

    def test_agent_treats_emergency_and_severe_patients_first(self) -> None:
        environment = HospitalEnvironment(seed=3)
        environment.waiting = [
            Patient(id=1, severity="high"),
            Patient(id=2, severity="high", emergency=True),
            Patient(id=3, severity="medium"),
        ]

        _, reward, _, info = environment.step(1)

        self.assertEqual(info["treated"], 1)
        self.assertEqual(info["reward_components"]["treated"], 3.0)
        self.assertEqual(info["reward_components"]["emergency_bonus"], 1.0)
        self.assertEqual(
            info["treated_patients"],
            [{"id": 2, "severity": "high", "emergency": True}],
        )
        self.assertEqual(reward, sum(info["reward_components"].values()))

    def test_waiting_patients_deteriorate_and_reward_is_penalized(self) -> None:
        environment = HospitalEnvironment(seed=8)
        environment.config = Task(
            beds=10,
            initial_patients=1,
            max_steps=2,
            arrival_rate=0.0,
        )
        environment.waiting = [Patient(id=1, severity="low")]

        with patch("rl_model.DETERIORATION_PROBABILITY", 1.0):
            state, reward, _, info = environment.step(0)

        patient = next(patient for patient in state["patients"] if patient["id"] == 1)
        self.assertEqual(patient["severity"], "medium")
        self.assertEqual(info["reward_components"]["deterioration"], -0.5)
        self.assertEqual(reward, -0.8)

    def test_emergency_arrival_has_twenty_percent_probability(self) -> None:
        self.assertEqual(EMERGENCY_ARRIVAL_PROBABILITY, 0.2)
        environment = HospitalEnvironment(seed=1)
        environment.config = Task(
            beds=10,
            initial_patients=0,
            max_steps=3,
            arrival_rate=0.0,
        )
        environment.waiting = []

        with patch("rl_model.EMERGENCY_ARRIVAL_PROBABILITY", 1.0):
            _, _, _, info = environment.step(0)

        self.assertEqual(info["emergency_arrivals"], 1)
        self.assertTrue(environment.state()["patients"][0]["emergency"])
        self.assertEqual(environment.state()["patients"][0]["severity"], "high")

    def test_unserved_high_and_emergency_patient_reward(self) -> None:
        environment = HospitalEnvironment(seed=1)
        environment.config = Task(
            beds=2,
            initial_patients=2,
            max_steps=2,
            arrival_rate=0.0,
        )
        environment.waiting = [
            Patient(id=1, severity="high"),
            Patient(id=2, severity="high", emergency=True),
        ]

        with patch("rl_model.EMERGENCY_ARRIVAL_PROBABILITY", 0.0):
            _, reward, _, info = environment.step(0)

        self.assertEqual(info["reward_components"]["untreated_high"], -4.0)
        self.assertEqual(info["reward_components"]["untreated_emergency"], -1.5)
        self.assertEqual(info["reward_components"]["wasted_bed"], -0.6)
        self.assertAlmostEqual(reward, -6.1)

    def test_action_validation_does_not_advance_episode(self) -> None:
        environment = HospitalEnvironment(seed=1)

        with self.assertRaises(ValueError):
            environment.step(environment.beds_available + 1)
        with self.assertRaises(ValueError):
            environment.step(-1)
        self.assertEqual(environment.step_number, 0)

    def test_occupied_beds_return_after_sampled_stay(self) -> None:
        environment = HospitalEnvironment(seed=2)
        environment.config = Task(
            beds=10,
            initial_patients=0,
            max_steps=3,
            arrival_rate=0.0,
        )
        environment.waiting = []
        environment.occupied = [Patient(id=1, severity="high", remaining_stay=2)]
        environment.rng = np.random.default_rng(10)

        _, _, _, first_info = environment.step(0)
        self.assertEqual(first_info["occupied_beds"], 1)
        _, _, _, second_info = environment.step(0)
        self.assertEqual(second_info["occupied_beds"], 0)

    def test_bed_freed_during_step_is_available_for_next_action(self) -> None:
        environment = HospitalEnvironment(seed=2)
        environment.config = Task(
            beds=2,
            initial_patients=1,
            max_steps=3,
            arrival_rate=0.0,
        )
        environment.waiting = [Patient(id=2, severity="high")]
        environment.occupied = [
            Patient(id=1, severity="high", remaining_stay=1)
        ]

        next_observation, _, _, _ = environment.step(0)

        self.assertEqual(next_observation["beds"], 2)
        self.assertEqual(
            list(QLearningAgent.valid_actions(next_observation)),
            [0, 1],
        )

    def test_discharged_beds_do_not_count_as_unavoidable_idle_beds(self) -> None:
        environment = HospitalEnvironment(seed=2)
        environment.config = Task(
            beds=2,
            initial_patients=2,
            max_steps=2,
            arrival_rate=0.0,
        )
        environment.waiting = [
            Patient(id=2, severity="high"),
            Patient(id=3, severity="medium"),
        ]
        environment.occupied = [
            Patient(id=1, severity="high", remaining_stay=1)
        ]

        with patch("rl_model.EMERGENCY_ARRIVAL_PROBABILITY", 0.0):
            _, _, _, info = environment.step(1)

        self.assertEqual(info["reward_components"]["wasted_bed"], 0.0)

    def test_grade_is_bounded_and_increases_with_return(self) -> None:
        environment = HospitalEnvironment(seed=1)
        initial_grade = environment.grade()
        environment.total_reward = 100.0
        positive_grade = environment.grade()
        expected_grade = 0.001 + 0.998 / (
            1 + np.exp(-100.0 / environment.config.max_steps)
        )
        environment.total_reward = -100.0
        negative_grade = environment.grade()

        self.assertAlmostEqual(positive_grade, expected_grade)
        self.assertGreater(positive_grade, initial_grade)
        self.assertLess(negative_grade, initial_grade)
        self.assertGreaterEqual(negative_grade, 0.001)
        self.assertLessEqual(positive_grade, 0.999)

    def test_q_learning_agent_trains_and_selects_a_valid_action(self) -> None:
        agent = QLearningAgent(training_episodes=20, seed=13)
        agent.train()
        observation = HospitalEnvironment(seed=14).state()

        action = agent.choose_action(observation)

        self.assertTrue(agent.trained)
        self.assertEqual(len(agent.episode_returns), 20)
        self.assertGreater(len(agent.q_table), 0)
        self.assertGreaterEqual(action, 0)
        self.assertLessEqual(
            action,
            min(observation["beds"], len(observation["patients"])),
        )

    def test_q_learning_training_is_repeatable_for_a_fixed_seed(self) -> None:
        first = QLearningAgent(training_episodes=5, seed=21)
        second = QLearningAgent(training_episodes=5, seed=21)

        first.train()
        second.train()

        self.assertEqual(first.episode_returns, second.episode_returns)
        self.assertEqual(dict(first.q_table), dict(second.q_table))

    def test_training_records_progress_metrics_and_frozen_trained_q_table(self) -> None:
        agent = QLearningAgent(training_episodes=5, seed=31)
        agent.train()

        self.assertEqual(len(agent.training_metrics), 5)
        self.assertEqual(agent.training_metrics[-1]["episode"], 5)
        self.assertEqual(agent.training_metrics[-1]["total_transitions"], 25)
        self.assertEqual(
            agent.training_metrics[-1]["learned_states"],
            len(agent.q_table),
        )
        self.assertEqual(
            agent.training_metrics[-1]["state_action_values"],
            sum(len(values) for values in agent.q_table.values()),
        )
        self.assertEqual(dict(agent.trained_q_table), dict(agent.q_table))
        self.assertGreater(len(agent.trained_q_table), 0)

    def test_execution_updates_q_values_without_changing_trained_snapshot(self) -> None:
        agent = QLearningAgent(training_episodes=5, seed=31)
        agent.train()
        observation = HospitalEnvironment(seed=agent.seed).state()
        state = agent.state_key(observation)
        action = next(iter(agent.trained_q_table[state]))
        trained_value = agent.trained_q_table[state][action]
        next_observation = {**observation, "done": True}

        updated_q, td_error = agent.learn_from_transition(
            observation,
            action,
            100.0,
            next_observation,
        )

        self.assertAlmostEqual(
            updated_q,
            trained_value + agent.learning_rate * (100.0 - trained_value),
        )
        self.assertNotEqual(updated_q, trained_value)
        self.assertEqual(agent.trained_q_table[state][action], trained_value)
        self.assertEqual(agent.q_table[state][action], updated_q)
        self.assertEqual(len(agent.execution_metrics), 1)
        self.assertEqual(agent.execution_metrics[0]["td_error"], td_error)

    def test_execution_update_rejects_invalid_action(self) -> None:
        agent = QLearningAgent(training_episodes=2, seed=12)
        observation = HospitalEnvironment(seed=14).state()

        with self.assertRaisesRegex(ValueError, "action must be valid"):
            agent.learn_from_transition(
                observation,
                observation["beds"] + 1,
                1.0,
                observation,
            )

    def test_execution_metrics_separate_reward_penalty_and_net_result(self) -> None:
        agent = QLearningAgent(training_episodes=1, seed=12)
        observation = HospitalEnvironment(seed=14).state()
        next_observation = {**observation, "done": True}

        updated_q, td_error = agent.learn_from_transition(
            observation,
            0,
            1.2,
            next_observation,
            reward_components={"treated": 1.5, "wasted_bed": -0.3},
        )

        metrics = agent.execution_metrics[0]
        self.assertAlmostEqual(metrics["gross_reward"], 1.5)
        self.assertAlmostEqual(metrics["penalty"], 0.3)
        self.assertAlmostEqual(metrics["final_result"], 1.2)
        self.assertAlmostEqual(updated_q, 0.18)
        self.assertAlmostEqual(td_error, 1.2)

class HospitalApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_reset_state_grade_and_step(self) -> None:
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

        reset = self.client.post("/reset?seed=12")
        self.assertEqual(reset.status_code, 200)
        self.assertEqual(reset.json()["beds"], 8)
        self.assertEqual(reset.json()["step"], 0)

        state = self.client.get("/state").json()
        self.assertEqual(len(state["patients"]), 8)

        grade = self.client.get("/grade").json()
        self.assertGreaterEqual(grade["grade"], 0.001)
        self.assertLessEqual(grade["grade"], 0.999)

        step = self.client.post("/step", json={"allocate": 1})
        self.assertEqual(step.status_code, 200)
        self.assertEqual(step.json()["observation"]["step"], 1)
        self.assertIn("reward_components", step.json()["info"])

    def test_api_rejects_invalid_actions(self) -> None:
        self.client.post("/reset")
        self.assertEqual(
            self.client.post("/step", json={"allocate": -1}).status_code,
            422,
        )
        self.assertEqual(
            self.client.post("/step", json={"allocate": 1.5}).status_code,
            422,
        )
        self.assertEqual(
            self.client.post("/step", json={"allocate": True}).status_code,
            422,
        )


if __name__ == "__main__":
    unittest.main()
