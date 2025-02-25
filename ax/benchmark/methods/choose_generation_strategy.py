# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from ax.benchmark.benchmark_method import (
    BenchmarkMethod,
    get_sequential_optimization_scheduler_options,
)
from ax.benchmark.benchmark_problem import BenchmarkProblemBase
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.service.scheduler import SchedulerOptions


def get_choose_generation_strategy_method(
    problem: BenchmarkProblemBase,
    scheduler_options: Optional[SchedulerOptions] = None,
    distribute_replications: bool = False,
) -> BenchmarkMethod:
    generation_strategy = choose_generation_strategy(
        search_space=problem.search_space,
        optimization_config=problem.optimization_config,
        num_trials=problem.num_trials,
    )

    return BenchmarkMethod(
        name=f"ChooseGenerationStrategy::{problem.name}",
        generation_strategy=generation_strategy,
        scheduler_options=scheduler_options
        or get_sequential_optimization_scheduler_options(),
        distribute_replications=distribute_replications,
    )
