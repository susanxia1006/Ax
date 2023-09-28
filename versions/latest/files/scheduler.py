from random import randint
from time import time
from typing import Any, Dict, NamedTuple, Union

from ax.core.base_trial import TrialStatus
from ax.utils.measurement.synthetic_functions import branin


class MockJob(NamedTuple):
    """Dummy class to represent a job scheduled on `MockJobQueue`."""

    id: int
    parameters: Dict[str, Union[str, float, int, bool]]


class MockJobQueueClient:
    """Dummy class to represent a job queue where the Ax `Scheduler` will
    deploy trial evaluation runs during optimization.
    """

    jobs: Dict[str, MockJob] = {}

    def schedule_job_with_parameters(
        self, parameters: Dict[str, Union[str, float, int, bool]]
    ) -> int:
        """Schedules an evaluation job with given parameters and returns job ID."""
        # Code to actually schedule the job and produce an ID would go here;
        # using timestamp in microseconds as dummy ID for this example.
        job_id = int(time() * 1e6)
        self.jobs[job_id] = MockJob(job_id, parameters)
        return job_id

    def get_job_status(self, job_id: int) -> TrialStatus:
        """ "Get status of the job by a given ID. For simplicity of the example,
        return an Ax `TrialStatus`.
        """
        job = self.jobs[job_id]
        # Instead of randomizing trial status, code to check actual job status
        # would go here.
        if randint(0, 3) > 0:
            return TrialStatus.COMPLETED
        return TrialStatus.RUNNING

    def get_outcome_value_for_completed_job(self, job_id: int) -> Dict[str, float]:
        """Get evaluation results for a given completed job."""
        job = self.jobs[job_id]
        # In a real external system, this would retrieve real relevant outcomes and
        # not a synthetic function value.
        return {"branin": branin(job.parameters.get("x1"), job.parameters.get("x2"))}


MOCK_JOB_QUEUE_CLIENT = MockJobQueueClient()


def get_mock_job_queue_client() -> MockJobQueueClient:
    """Obtain the singleton job queue instance."""
    return MOCK_JOB_QUEUE_CLIENT

from collections import defaultdict
from typing import Iterable, Set

from ax.core.base_trial import BaseTrial
from ax.core.runner import Runner
from ax.core.trial import Trial


class MockJobRunner(Runner):  # Deploys trials to external system.
    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        """Deploys a trial based on custom runner subclass implementation.

        Args:
            trial: The trial to deploy.

        Returns:
            Dict of run metadata from the deployment process.
        """
        if not isinstance(trial, Trial):
            raise ValueError("This runner only handles `Trial`.")

        mock_job_queue = get_mock_job_queue_client()
        job_id = mock_job_queue.schedule_job_with_parameters(
            parameters=trial.arm.parameters
        )
        # This run metadata will be attached to trial as `trial.run_metadata`
        # by the base `Scheduler`.
        return {"job_id": job_id}

    def poll_trial_status(
        self, trials: Iterable[BaseTrial]
    ) -> Dict[TrialStatus, Set[int]]:
        """Checks the status of any non-terminal trials and returns their
        indices as a mapping from TrialStatus to a list of indices. Required
        for runners used with Ax ``Scheduler``.

        NOTE: Does not need to handle waiting between polling calls while trials
        are running; this function should just perform a single poll.

        Args:
            trials: Trials to poll.

        Returns:
            A dictionary mapping TrialStatus to a list of trial indices that have
            the respective status at the time of the polling. This does not need to
            include trials that at the time of polling already have a terminal
            (ABANDONED, FAILED, COMPLETED) status (but it may).
        """
        status_dict = defaultdict(set)
        for trial in trials:
            mock_job_queue = get_mock_job_queue_client()
            status = mock_job_queue.get_job_status(
                job_id=trial.run_metadata.get("job_id")
            )
            status_dict[status].add(trial.index)

        return status_dict

import pandas as pd

from ax.core.metric import Metric, MetricFetchResult, MetricFetchE
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.utils.common.result import Ok, Err


class BraninForMockJobMetric(Metric):  # Pulls data for trial from external system.
    def fetch_trial_data(self, trial: BaseTrial) -> MetricFetchResult:
        """Obtains data via fetching it from ` for a given trial."""
        if not isinstance(trial, Trial):
            raise ValueError("This metric only handles `Trial`.")

        try:
            mock_job_queue = get_mock_job_queue_client()

            # Here we leverage the "job_id" metadata created by `MockJobRunner.run`.
            branin_data = mock_job_queue.get_outcome_value_for_completed_job(
                job_id=trial.run_metadata.get("job_id")
            )
            df_dict = {
                "trial_index": trial.index,
                "metric_name": "branin",
                "arm_name": trial.arm.name,
                "mean": branin_data.get("branin"),
                # Can be set to 0.0 if function is known to be noiseless
                # or to an actual value when SEM is known. Setting SEM to
                # `None` results in Ax assuming unknown noise and inferring
                # noise level from data.
                "sem": None,
            }
            return Ok(value=Data(df=pd.DataFrame.from_records([df_dict])))
        except Exception as e:
            return Err(
                MetricFetchE(message=f"Failed to fetch {self.name}", exception=e)
            )

from ax import *


def make_branin_experiment_with_runner_and_metric() -> Experiment:
    parameters = [
        RangeParameter(
            name="x1",
            parameter_type=ParameterType.FLOAT,
            lower=-5,
            upper=10,
        ),
        RangeParameter(
            name="x2",
            parameter_type=ParameterType.FLOAT,
            lower=0,
            upper=15,
        ),
    ]

    objective = Objective(metric=BraninForMockJobMetric(name="branin"), minimize=True)

    return Experiment(
        name="branin_test_experiment",
        search_space=SearchSpace(parameters=parameters),
        optimization_config=OptimizationConfig(objective=objective),
        runner=MockJobRunner(),
        is_test=True,  # Marking this experiment as a test experiment.
    )


experiment = make_branin_experiment_with_runner_and_metric()

from ax.modelbridge.dispatch_utils import choose_generation_strategy

generation_strategy = choose_generation_strategy(
    search_space=experiment.search_space,
    max_parallelism_cap=3,
)

from ax.service.scheduler import Scheduler, SchedulerOptions


scheduler = Scheduler(
    experiment=experiment,
    generation_strategy=generation_strategy,
    options=SchedulerOptions(),
)



import numpy as np
from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()


def get_plot():
    best_objectives = np.array(
        [[trial.objective_mean for trial in scheduler.experiment.trials.values()]]
    )
    best_objective_plot = optimization_trace_single_method(
        y=np.minimum.accumulate(best_objectives, axis=1),
        title="Model performance vs. # of iterations",
        ylabel="Y",
    )
    return best_objective_plot

scheduler.run_n_trials(max_trials=3)

best_objective_plot = get_plot()
render(best_objective_plot)

from ax.service.utils.report_utils import exp_to_df

exp_to_df(experiment)

scheduler.run_n_trials(max_trials=3)

best_objective_plot = get_plot()
render(best_objective_plot)

exp_to_df(experiment)

scheduler.run_n_trials(max_trials=3, timeout_hours=0.00001)

best_objective_plot = get_plot()
render(best_objective_plot)

from ax.storage.registry_bundle import RegistryBundle
from ax.storage.sqa_store.db import (
    create_all_tables,
    get_engine,
    init_engine_and_session_factory,
)
from ax.storage.sqa_store.decoder import Decoder
from ax.storage.sqa_store.encoder import Encoder
from ax.storage.sqa_store.sqa_config import SQAConfig
from ax.storage.sqa_store.structs import DBSettings

bundle = RegistryBundle(
    metric_clss={BraninForMockJobMetric: None}, runner_clss={MockJobRunner: None}
)

# URL is of the form "dialect+driver://username:password@host:port/database".
# Instead of URL, can provide a `creator function`; can specify custom encoders/decoders if necessary.
db_settings = DBSettings(
    url="sqlite:///foo.db",
    encoder=bundle.encoder,
    decoder=bundle.decoder,
)

# The following lines are only necessary because it is the first time we are using this database
# in practice, you will not need to run these lines every time you initialize your scheduler
init_engine_and_session_factory(url=db_settings.url)
engine = get_engine()
create_all_tables(engine)

stored_experiment = make_branin_experiment_with_runner_and_metric()
generation_strategy = choose_generation_strategy(search_space=experiment.search_space)

scheduler_with_storage = Scheduler(
    experiment=stored_experiment,
    generation_strategy=generation_strategy,
    options=SchedulerOptions(),
    db_settings=db_settings,
)

reloaded_experiment_scheduler = Scheduler.from_stored_experiment(
    experiment_name="branin_test_experiment",
    options=SchedulerOptions(),
    # `DBSettings` are also required here so scheduler has access to the
    # database, from which it needs to load the experiment.
    db_settings=db_settings,
)

reloaded_experiment_scheduler.run_n_trials(max_trials=3)

print(SchedulerOptions.__doc__)

class ResultReportingScheduler(Scheduler):
    def report_results(self, force_refit: bool = False):
        return True, {
            "trials so far": len(self.experiment.trials),
            "currently producing trials from generation step": self.generation_strategy._curr.model_name,
            "running trials": [t.index for t in self.running_trials],
        }

experiment = make_branin_experiment_with_runner_and_metric()
scheduler = ResultReportingScheduler(
    experiment=experiment,
    generation_strategy=choose_generation_strategy(
        search_space=experiment.search_space,
        max_parallelism_cap=3,
    ),
    options=SchedulerOptions(),
)

for reported_result in scheduler.run_trials_and_yield_results(max_trials=6):
    print("Reported result: ", reported_result)

# Clean up to enable running the tutorial repeatedly with
# the same results. You wouldn't do this if you wanted to
# keep adding data to the same experiment.
from ax.storage.sqa_store.delete import delete_experiment

delete_experiment("branin_test_experiment")


