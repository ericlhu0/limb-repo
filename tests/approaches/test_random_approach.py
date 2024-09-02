"""Tests for random_approach.py."""

import numpy as np

from python_research_starter.approaches.random_approach import RandomApproach
from python_research_starter.benchmarks.maze_benchmark import MazeBenchmark


def test_random_approach():
    """Tests for RandomApproach()."""
    benchmark = MazeBenchmark()
    approach = RandomApproach(
        benchmark.get_actions(),
        benchmark.get_next_state,
        benchmark.get_cost,
        benchmark.check_goal,
    )
    assert approach.get_name() == "random"
    rng = np.random.default_rng(123)
    task = benchmark.generate_tasks(1, "train", rng)[0]
    plan = approach.generate_plan(task, "test", 1e-1, rng)
    assert len(plan) > 0
