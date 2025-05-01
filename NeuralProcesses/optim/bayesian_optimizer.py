"""
This module contains the :class:`BayesianOptimizer` class, used to perform Bayesian optimization.
"""

from __future__ import annotations

import copy
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Mapping,
    MutableMapping,
    Optional,
    TypeVar,
    cast,
    overload,
)
from ..models import MultiTimeSeriesInputModel

import absl
import dill
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist

from .acquisition.function.function import non_dominated
from .models.utils import optimize_model_and_save_result

try:
    import pandas as pd
    import seaborn as sns
except ModuleNotFoundError:
    pd = None
    sns = None

from . import logging
from .acquisition.function.function import (
    AcquisitionRule,
    EfficientGlobalOptimization,
    LocalDatasetsAcquisitionRule,
)
from .acquisition.function.function import with_local_datasets
from .data import Dataset
from .models import (
    ProbabilisticModel,
    SupportsCovarianceWithTopFidelity,
    TrainableProbabilisticModel,
)
from .objectives.utils import mk_batch_observer
from .observer import OBJECTIVE, Observer
from .space import SearchSpace
from .types import State, Tag, TensorType
from .utils import Err, Ok, Result, Timer
from .utils.misc import LocalizedTag, get_value_for_tag, ignoring_local_tags

from flax.training import TrainState
StateType = TypeVar("StateType")
""" Unbound type variable. """

SearchSpaceType = TypeVar("SearchSpaceType", bound=SearchSpace)
""" Type variable bound to :class:`SearchSpace`. """

ProbabilisticModelType = TypeVar(
    "ProbabilisticModelType",
    bound=ProbabilisticModel,
    covariant=True,
)
""" Covariant type variable bound to :class:`ProbabilisticModel`. """

TrainableProbabilisticModelType = TypeVar(
    "TrainableProbabilisticModelType", bound=TrainableProbabilisticModel, contravariant=True
)
""" Contravariant type variable bound to :class:`TrainableProbabilisticModel`. """

EarlyStopCallback = Callable[
    [Mapping[Tag, Dataset], Mapping[Tag, TrainableProbabilisticModelType], Optional[StateType]],
    bool,
]
""" Early stop callback type, generic in the model and state types. """


@dataclass(frozen=True)
class Record(Generic[StateType, ProbabilisticModelType]):
    """Container to record the state of each step of the optimization process."""

    datasets: Mapping[Tag, Dataset]
    """ The known data from the observer. """

    models: Mapping[Tag, ProbabilisticModelType]
    """ The models over the :attr:`datasets`. """

    acquisition_state: StateType | None
    """ The acquisition state. """

    @property
    def dataset(self) -> Dataset:
        """The dataset when there is just one dataset."""
        # Ignore local datasets.
        datasets: Mapping[Tag, Dataset] = ignoring_local_tags(self.datasets)
        if len(datasets) == 1:
            return next(iter(datasets.values()))
        else:
            raise ValueError(f"Expected a single dataset, found {len(datasets)}")

    @property
    def model(self) -> ProbabilisticModelType:
        """The model when there is just one dataset."""
        # Ignore local models.
        models: Mapping[Tag, ProbabilisticModelType] = ignoring_local_tags(self.models)
        if len(models) == 1:
            return next(iter(models.values()))
        else:
            raise ValueError(f"Expected a single model, found {len(models)}")

    def save(self, path: Path | str) -> FrozenRecord[StateType, ProbabilisticModelType]:
        """Save the record to disk. Will overwrite any existing file at the same path."""
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)
        return FrozenRecord(Path(path))


@dataclass(frozen=True)
class FrozenRecord(Generic[StateType, ProbabilisticModelType]):
    """
    A Record container saved on disk.

    Note that records are saved via pickling and are therefore neither portable nor secure.
    Only open frozen records generated on the same system.
    """

    path: Path
    """ The path to the pickled Record. """

    def load(self) -> Record[StateType, ProbabilisticModelType]:
        """Load the record into memory."""
        with open(self.path, "rb") as f:
            return dill.load(f)

    @property
    def datasets(self) -> Mapping[Tag, Dataset]:
        """The known data from the observer."""
        return self.load().datasets

    @property
    def models(self) -> Mapping[Tag, ProbabilisticModelType]:
        """The models over the :attr:`datasets`."""
        return self.load().models

    @property
    def acquisition_state(self) -> StateType | None:
        """The acquisition state."""
        return self.load().acquisition_state

    @property
    def dataset(self) -> Dataset:
        """The dataset when there is just one dataset."""
        return self.load().dataset

    @property
    def model(self) -> ProbabilisticModelType:
        """The model when there is just one dataset."""
        return self.load().model


# this should be a generic NamedTuple, but mypy doesn't support them
#  https://github.com/python/mypy/issues/685
@dataclass(frozen=True)
class OptimizationResult(Generic[StateType, ProbabilisticModelType]):
    """The final result, and the historical data of the optimization process."""

    final_result: Result[Record[StateType, ProbabilisticModelType]]
    """
    The final result of the optimization process. This contains either a :class:`Record` or an
    exception.
    """

    history: list[
        Record[StateType, ProbabilisticModelType] | FrozenRecord[StateType, ProbabilisticModelType]
    ]
    r"""
    The history of the :class:`Record`\ s from each step of the optimization process. These
    :class:`Record`\ s are created at the *start* of each loop, and as such will never
    include the :attr:`final_result`. The records may be either in memory or on disk.
    """

    @staticmethod
    def step_filename(step: int, num_steps: int) -> str:
        """Default filename for saved optimization steps."""
        return f"step.{step:0{len(str(num_steps - 1))}d}.pickle"

    STEP_GLOB: ClassVar[str] = "step.*.pickle"
    RESULTS_FILENAME: ClassVar[str] = "results.pickle"

    def astuple(
        self,
    ) -> tuple[
        Result[Record[StateType, ProbabilisticModelType]],
        list[
            Record[StateType, ProbabilisticModelType]
            | FrozenRecord[StateType, ProbabilisticModelType]
        ],
    ]:
        """
        **Note:** In contrast to the standard library function :func:`dataclasses.astuple`, this
        method does *not* deepcopy instance attributes.

        :return: The :attr:`final_result` and :attr:`history` as a 2-tuple.
        """
        return self.final_result, self.history

    @property
    def is_ok(self) -> bool:
        """`True` if the final result contains a :class:`Record`."""
        return self.final_result.is_ok

    @property
    def is_err(self) -> bool:
        """`True` if the final result contains an exception."""
        return self.final_result.is_err

    def try_get_final_datasets(self) -> Mapping[Tag, Dataset]:
        """
        Convenience method to attempt to get the final data.

        :return: The final data, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        """
        return self.final_result.unwrap().datasets

    def try_get_final_dataset(self) -> Dataset:
        """
        Convenience method to attempt to get the final data for a single dataset run.

        :return: The final data, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        :raise ValueError: If the optimization was not a single dataset run.
        """
        datasets = self.try_get_final_datasets()
        # Ignore local datasets.
        datasets = ignoring_local_tags(datasets)
        if len(datasets) == 1:
            return next(iter(datasets.values()))
        else:
            raise ValueError(f"Expected a single dataset, found {len(datasets)}")

    def try_get_optimal_point(self) -> tuple[TensorType, TensorType, TensorType]:
        """
        Convenience method to attempt to get the optimal point for a single dataset,
        single objective run.

        :return: Tuple of the optimal query point, observation and its index.
        """
        dataset = self.try_get_final_dataset()
        if tf.rank(dataset.observations) != 2 or dataset.observations.shape[1] != 1:
            raise ValueError("Expected a single objective")
        if tf.reduce_any(
            [
                isinstance(model, SupportsCovarianceWithTopFidelity)
                for model in self.try_get_final_models()
            ]
        ):
            raise ValueError("Expected single fidelity models")
        arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
        return dataset.query_points[arg_min_idx], dataset.observations[arg_min_idx], arg_min_idx

    def try_get_final_models(self) -> Mapping[Tag, ProbabilisticModelType]:
        """
        Convenience method to attempt to get the final models.

        :return: The final models, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        """
        return self.final_result.unwrap().models

    def try_get_final_model(self) -> ProbabilisticModelType:
        """
        Convenience method to attempt to get the final model for a single model run.

        :return: The final model, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        :raise ValueError: If the optimization was not a single model run.
        """
        models = self.try_get_final_models()
        # Ignore local models.
        models = ignoring_local_tags(models)
        if len(models) == 1:
            return next(iter(models.values()))
        else:
            raise ValueError(f"Expected single model, found {len(models)}")

    @property
    def loaded_history(self) -> list[Record[StateType, ProbabilisticModelType]]:
        """The history of the optimization process loaded into memory."""
        return [record if isinstance(record, Record) else record.load() for record in self.history]

    def save_result(self, path: Path | str) -> None:
        """Save the final result to disk. Will overwrite any existing file at the same path."""
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            dill.dump(self.final_result, f, dill.HIGHEST_PROTOCOL)

    def save(self, base_path: Path | str) -> None:
        """Save the optimization result to disk. Will overwrite existing files at the same path."""
        path = Path(base_path)
        num_steps = len(self.history)
        self.save_result(path / self.RESULTS_FILENAME)
        for i, record in enumerate(self.loaded_history):
            record_path = path / self.step_filename(i, num_steps)
            record.save(record_path)

    @classmethod
    def from_path(
        cls, base_path: Path | str
    ) -> OptimizationResult[StateType, ProbabilisticModelType]:
        """Load a previously saved OptimizationResult."""
        try:
            with open(Path(base_path) / cls.RESULTS_FILENAME, "rb") as f:
                result = dill.load(f)
        except FileNotFoundError as e:
            result = Err(e)

        history: list[
            Record[StateType, ProbabilisticModelType]
            | FrozenRecord[StateType, ProbabilisticModelType]
        ] = [FrozenRecord(file) for file in sorted(Path(base_path).glob(cls.STEP_GLOB))]
        return cls(result, history)


class ODEOptimizer(Generic[SearchSpaceType]):
    """
    This class performs optimization, the data-efficient optimization of an expensive
    black-box *objective function* over some *search space*. Since we may not have access to the
    objective function itself, we speak instead of an *observer* that observes it.
    """

    def __init__(self, observer: Observer, search_space: SearchSpaceType):
        """
        :param observer: The observer of the objective function.
        :param search_space: The space over which to search. Must be a
            :class:`~trieste.space.SearchSpace`.
        """
        self._observer = observer
        self._search_space = search_space


    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        model: MultiTimeSeriesInputModel,
        model_state: TrainState, 
        acquisition_rule: AcquisitionRule[
            TensorType | State[StateType | None, TensorType],
            SearchSpaceType,
        ]
        | None = None,
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_model: bool = True,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, StateType]
        ] = None,
        start_step: int = 0,
    ) -> (
        OptimizationResult[StateType, TrainableProbabilisticModelType]
        | OptimizationResult[None, TrainableProbabilisticModelType]
    ):
        """
        Attempt to find the minimizer of the ``observer`` in the ``search_space`` (both specified at
        :meth:`__init__`). This is the central implementation of the Bayesian optimization loop.

        For each step in ``num_steps``, this method:
            - Finds the next points with which to query the ``observer`` using the
              ``acquisition_rule``'s :meth:`acquire` method, passing it the ``search_space``,
              ``datasets``, ``models``, and current acquisition state.
            - Queries the ``observer`` *once* at those points.
            - Updates the datasets and models with the data from the ``observer``.

        If any errors are raised during the optimization loop, this method will catch and return
        them instead and print a message (using `absl` at level `absl.logging.ERROR`).
        If ``track_state`` is enabled, then in addition to the final result, the history of the
        optimization process will also be returned. If ``track_path`` is also set, then
        the history and final result will be saved to disk rather than all being kept in memory.

        **Type hints:**
            - The ``acquisition_rule`` must use the same type of
              :class:`~trieste.space.SearchSpace` as specified in :meth:`__init__`.
            - The ``acquisition_state`` must be of the type expected by the ``acquisition_rule``.
              Any acquisition state in the optimization result will also be of this type.

        :param num_steps: The number of optimization steps to run.
        :param datasets: The known observer query points and observations for each tag.
        :param models: The model to use for each :class:`~trieste.data.Dataset` in
            ``datasets``.
        :param acquisition_rule: The acquisition rule, which defines how to search for a new point
            on each optimization step. Defaults to
            :class:`~trieste.acquisition.rule.EfficientGlobalOptimization` with default
            arguments. Note that if the default is used, this implies the tags must be
            `OBJECTIVE`, the search space can be any :class:`~trieste.space.SearchSpace`, and the
            acquisition state returned in the :class:`OptimizationResult` will be `None`.
        :param acquisition_state: The acquisition state to use on the first optimization step.
            This argument allows the caller to restore the optimization process from an existing
            :class:`Record`.
        :param track_state: If `True`, this method saves the optimization state at the start of each
            step. Models and acquisition state are copied using `copy.deepcopy`.
        :param track_path: If set, the optimization state is saved to disk at this path,
            rather than being copied in memory.
        :param fit_model: If `False` then we never fit the model during BO (e.g. if we
            are using a rule that doesn't rely on the models and don't want to waste computation).
        :param fit_initial_model: If `False` then we assume that the initial models have
            already been optimized on the datasets and so do not require optimization before
            the first optimization step.
        :param early_stop_callback: An optional callback that is evaluated with the current
            datasets, models and optimization state before every optimization step. If this
            returns `True` then the optimization loop is terminated early.
        :param start_step: The step number to start with. This number is removed from ``num_steps``
            and is useful for restarting previous computations.
        :return: An :class:`OptimizationResult`. The :attr:`final_result` element contains either
            the final optimization data, models and acquisition state, or, if an exception was
            raised while executing the optimization loop, it contains the exception raised. In
            either case, the :attr:`history` element is the history of the data, models and
            acquisition state at the *start* of each optimization step (up to and including any step
            that fails to complete). The history will never include the final optimization result.
        :raise ValueError: If any of the following are true:

            - ``num_steps`` is negative.
            - the keys in ``datasets`` and ``models`` do not match
            - ``datasets`` or ``models`` are empty
            - the default `acquisition_rule` is used and the tags are not `OBJECTIVE`.
        """
        # Copy the dataset so we don't change the one provided by the user.
        datasets = copy.deepcopy(datasets)

        filtered_datasets = datasets

        if num_steps < 0:
            raise ValueError(f"num_steps must be at least 0, got {num_steps}")

        if not datasets:
            raise ValueError("dicts of datasets and models must be populated.")

        if acquisition_rule is None:
            if datasets.keys() != {OBJECTIVE}:
                raise ValueError(
                    f"Default acquisition rule EfficientGlobalOptimization requires tag"
                    f" {OBJECTIVE!r}, got keys {datasets.keys()}"
                )

            acquisition_rule = EfficientGlobalOptimization[
                SearchSpaceType, TrainableProbabilisticModelType
            ]()

        history: list[
            FrozenRecord[StateType, TrainableProbabilisticModelType]
            | Record[StateType, TrainableProbabilisticModelType]
        ] = []
        
        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=0):
                write_summary_init(
                    self._observer,
                    self._search_space,
                    acquisition_rule,
                    datasets,
                    model,
                    num_steps,
                )


        for step in range(start_step + 1, num_steps + 1):
            logging.set_step_number(step)

            try:
                with Timer() as total_step_wallclock_timer:
                    with Timer() as query_point_generation_timer:
                        # TODO: 
                        points_or_stateful = acquisition_rule.acquire(
                            self._search_space, model, datasets=datasets
                        )
                        query_points = points_or_stateful

                    observer = self._observer
                    observer_output = observer(query_points)

                    # TODO: 
                    for tag, new_dataset in tagged_output.items():
                        datasets[tag] += new_dataset

                if summary_writer:
                    with summary_writer.as_default(step=step):
                        write_summary_observations(
                            datasets,
                            model,
                            tagged_output,
                            model_fitting_timer,
                            observation_plot_dfs,
                        )
                        write_summary_query_points(
                            datasets,
                            model,
                            self._search_space,
                            query_points,
                            query_point_generation_timer,
                            query_plot_dfs,
                        )
                        logging.scalar("wallclock/step", total_step_wallclock_timer.time)

            except Exception as error:  # pylint: disable=broad-except
                tf.print(
                    f"\nOptimization failed at step {step}, encountered error with traceback:"
                    f"\n{traceback.format_exc()}"
                    f"\nTerminating optimization and returning the optimization history. You may "
                    f"be able to use the history to restart the process from a previous successful "
                    f"optimization step.\n",
                    output_stream=absl.logging.ERROR,
                )
                if isinstance(error, MemoryError):
                    tf.print(
                        "\nOne possible cause of memory errors is trying to evaluate acquisition "
                        "\nfunctions over large datasets, e.g. when initializing optimizers. "
                        "\nYou may be able to word around this by splitting up the evaluation "
                        "\nusing split_acquisition_function or split_acquisition_function_calls.",
                        output_stream=absl.logging.ERROR,
                    )
                result = OptimizationResult(Err(error), history)
                if track_state and track_path is not None:
                    result.save_result(Path(track_path) / OptimizationResult.RESULTS_FILENAME)
                return result

        tf.print("Optimization completed without errors", output_stream=absl.logging.INFO)

        record = Record(datasets, model, acquisition_state)
        result = OptimizationResult(Ok(record), history)
        if track_state and track_path is not None:
            result.save_result(Path(track_path) / OptimizationResult.RESULTS_FILENAME)
        return result
    

def write_summary_init(
    observer: Observer,
    search_space: SearchSpace,
    acquisition_rule: AcquisitionRule[
        TensorType | State[StateType | None, TensorType],
        SearchSpaceType,
        TrainableProbabilisticModelType,
    ],
    datasets: Mapping[Tag, Dataset],
    models: Mapping[Tag, TrainableProbabilisticModel],
    num_steps: int,
) -> None:
    """Write initial BO loop TensorBoard summary."""
    devices = tf.config.list_logical_devices()
    logging.text(
        "metadata",
        f"Observer: `{observer}`\n\n"
        f"Number of steps: `{num_steps}`\n\n"
        f"Number of initial points: "
        f"`{dict((k, len(v)) for k, v in datasets.items())}`\n\n"
        f"Search Space: `{search_space}`\n\n"
        f"Acquisition rule:\n\n    {acquisition_rule}\n\n"
        f"Models:\n\n    {models}\n\n"
        f"Available devices: `{dict(Counter(d.device_type for d in devices))}`",
    )

