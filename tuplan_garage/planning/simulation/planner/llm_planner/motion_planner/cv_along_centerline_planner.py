from typing import List, Tuple
from torch import Tensor
import numpy as np
from shapely import Point

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.training.preprocessing.features.trajectory_utils import _convert_absolute_to_relative_states
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from tuplan_garage.planning.simulation.planner.llm_planner.motion_planner.abstract_motion_planner import (
    AbstractMotionPlanner,
    modify_heading_of_current_state
)


class CVAlongCenterlinePlanner(AbstractMotionPlanner):
    def __init__(self):
        pass

    def initialize(self, initialization: PlannerInitialization) -> None:
        pass

    def compute_motion_trajectory(
        self,
        current_input: PlannerInput,
        target_centerline: List[StateSE2],
        centerline_offset: float,
        speed_limit_mps: float,
    ) -> AbstractTrajectory:

        # transform centerline from global to local coordinates
        local_centerline_poses = _convert_absolute_to_relative_states(
            origin_absolute_state=current_input.history.current_state[0].rear_axle,
            absolute_states=target_centerline
        )
        target_centerline = [(state.x, state.y, state.heading) for state in local_centerline_poses]

        predictions = self.constant_velocity_along_centerline(
            centerline=target_centerline,
            velocity=current_input.history.ego_states[0].dynamic_car_state.speed
        )
        # Convert relative poses to absolute states and wrap in a trajectory object.
        states = transform_predictions_to_states(
            predictions.data, current_input.history.ego_states, 8.0, 0.5
        )
        # rotate current state s.t. it points to the first future state
        states = modify_heading_of_current_state(states)
        return InterpolatedTrajectory(states)

    def constant_velocity_along_centerline(
        self,
        centerline=List[List[Tensor]],
        velocity=float
    ) -> Trajectory:
        path = PDMPath(
            discrete_path=[
                StateSE2(
                    x=pose[0],
                    y=pose[1],
                    heading=pose[2],
                    )
                for pose in centerline
            ]
        )
        starting_distance = path.project(Point(0,0))
        cv_states: List[StateSE2] = path.interpolate(
            distances=[velocity*0.5*(t+1) + starting_distance for t in range(16)]
        )
        return Trajectory(
            data=np.array(
                [
                    [pose.x, pose.y, pose.heading]
                    for pose in cv_states
                ]
            )
        )