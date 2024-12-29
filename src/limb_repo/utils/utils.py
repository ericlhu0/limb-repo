"""Utility functions."""

import os

from limb_repo.benchmarks.base_benchmark import Benchmark
from limb_repo.structs import Action, Task

def get_root_path() -> str:
    """Get the root path of the repository."""
    return os.path.abspath(os.path.join(__file__, "../../../..")
)

def to_abs_path(input_path: str) -> str:
    """Get the absolute path of the repository."""
    return os.path.abspath(os.path.join(get_root_path(), input_path))

def plan_is_valid(plan: list[Action], task: Task, benchmark: Benchmark) -> bool:
    """Checks if the plan solves the task."""
    state = task.init
    for action in plan:
        state = benchmark.get_next_state(state, action)
    return benchmark.check_goal(state, task.goal)


def get_plan_cost(plan: list[Action], task: Task, benchmark: Benchmark) -> float:
    """Get the total plan cost."""
    cost = 0.0
    state = task.init
    for action in plan:
        next_state = benchmark.get_next_state(state, action)
        cost += benchmark.get_cost(state, action, next_state)
        state = next_state
    return cost

if __name__ == "__main__":
    print(get_root_path())
    print("Done.")