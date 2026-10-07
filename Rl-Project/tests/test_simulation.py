import unittest

from rl_model import HospitalSimulator, PolicySolver, SimulationConfig


class HospitalSimulationTests(unittest.TestCase):
    def test_policy_moves_beds_toward_higher_covid_demand(self) -> None:
        config = SimulationConfig(normal_arrivals=1.0, covid_arrivals=6.0)
        solver = PolicySolver(config)

        self.assertGreater(solver.action_for((5, 5, 5)), 0)

    def test_simulation_is_repeatable_and_preserves_bed_capacity(self) -> None:
        config = SimulationConfig(simulation_days=12, seed=31)
        solver = PolicySolver(config)
        first = HospitalSimulator(config)
        second = HospitalSimulator(config)

        for _ in range(config.simulation_days):
            first.step(solver)
            second.step(solver)
            self.assertEqual(first.history[-1], second.history[-1])
            self.assertEqual(
                first.normal_capacity + first.covid_capacity,
                config.total_beds,
            )
            self.assertLessEqual(first.normal_available, first.normal_capacity)
            self.assertLessEqual(first.covid_available, first.covid_capacity)

        self.assertTrue(first.finished)

    def test_invalid_capacity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SimulationConfig(total_beds=1).validate()


if __name__ == "__main__":
    unittest.main()
