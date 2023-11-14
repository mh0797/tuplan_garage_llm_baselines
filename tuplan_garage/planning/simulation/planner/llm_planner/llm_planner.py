import time
from typing import List, Optional, Type, Tuple

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from tuplan_garage.planning.simulation.planner.llm_planner.behavior_planner.abstract_llm_behavior_planner import AbstractLLMBehaviorPlanner
from tuplan_garage.planning.training.preprocessing.feature_builders.llm_feature_builder import LLMFeatureBuilder
from tuplan_garage.planning.training.preprocessing.features.llm_features import LLMFeatures
from tuplan_garage.planning.simulation.planner.llm_planner.motion_planner.abstract_motion_planner import AbstractMotionPlanner

def relative_centerline_poses_to_absolute_states(
    relative_centerline_poses: List[Tuple[float,float,float]],
    origin_absolute_pose: StateSE2,
    ) -> List[StateSE2]:
    relative_centerline_poses = [
        StateSE2(x=pose[0], y=pose[1], heading=pose[2])
        for pose in relative_centerline_poses
    ]
    return relative_to_absolute_poses(
        origin_pose=origin_absolute_pose,
        relative_poses=relative_centerline_poses
    )

class LLMPlanner(AbstractPlanner):
    """
    Implements abstract planner interface.
    Used for simulating the PDM+LLM planner
    """

    def __init__(
        self,
        llm_feature_builder: LLMFeatureBuilder,
        llm_behavior_planner: AbstractLLMBehaviorPlanner,
        motion_planner: AbstractMotionPlanner,
        infer_llm_every_n_iterations: int,
        ) -> None:
        """
        Initializes the LLM_PDM planner class.
        :param model: Model to use for inference.
        """
        # self._model_loader = ModelLoader(model)
        self.llm_feature_builder = llm_feature_builder
        
        self._initialization: Optional[PlannerInitialization] = None

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

        self._llm_behavior_planner = llm_behavior_planner
        self._motion_planner = motion_planner
        
        self._infer_llm_every_n_iterations = infer_llm_every_n_iterations

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._initialization = initialization
        self._motion_planner.initialize(initialization)
        self._llm_behavior_planner.initialize(initialization)
        self._iteration = 0

    def get_comment_on_current_behavior(self) -> str:
        return self._llm_behavior_planner.get_comment_on_current_behavior()
    
    def get_prompt_for_current_behavior(self) -> str:
        return self._llm_behavior_planner.get_prompt_for_current_behavior()

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        if self._iteration % self._infer_llm_every_n_iterations == 0:
            # generate LLM features
            start_time = time.perf_counter()
            llm_features: LLMFeatures = self.llm_feature_builder.get_features_from_simulation(
                current_input=current_input,
                initialization=self._initialization
                )
            self._feature_building_runtimes.append(time.perf_counter() - start_time)

            # infer LLM
            start_time = time.perf_counter()
            response = self._llm_behavior_planner.infer_behavior(llm_features)
            self._current_target_centerline = relative_centerline_poses_to_absolute_states(
                relative_centerline_poses=response.centerline,
                origin_absolute_pose=current_input.history.current_state[0].center,
            )
            self._current_centerline_offset = response.centerline_offset
            self._current_speed_limit_mps = response.speed_limit
            self._inference_runtimes.append(time.perf_counter() - start_time)

        trajectory = self._motion_planner.compute_motion_trajectory(
            current_input=current_input,
            target_centerline=self._current_target_centerline,
            centerline_offset=self._current_centerline_offset,
            speed_limit_mps=self._current_speed_limit_mps,
        )

        self._iteration += 1
        
        return trajectory

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []
        return report