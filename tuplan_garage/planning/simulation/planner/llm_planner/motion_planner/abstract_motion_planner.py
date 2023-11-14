from abc import abstractmethod
from typing import List, Tuple
import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput
)

def modify_heading_of_current_state(
    states: List[EgoState],
    current_state: EgoState=None,
    point_to_n_th_state: int=1,
     ) -> List[EgoState]:
    '''
    Takes a trajectory of future states (including the current state) and adjusts the
    heading of the current state, so that it points to the point_to_n_th_state.
    This way, the controller encounters an inmediate heading offset,
    resulting in higher steering angle than if the current state just points forward
    '''
    if current_state is None:
        current_state = states[0]
    modified_heading = np.arctan2(
        states[point_to_n_th_state].rear_axle.y - current_state.rear_axle.y,
        states[point_to_n_th_state].rear_axle.x - current_state.rear_axle.x
    )
    modified_current_state = EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(x=current_state.rear_axle.x, y=current_state.rear_axle.y, heading=modified_heading),
        rear_axle_velocity_2d=current_state.dynamic_car_state.rear_axle_velocity_2d,
        rear_axle_acceleration_2d=current_state.dynamic_car_state.rear_axle_acceleration_2d,
        tire_steering_angle=current_state.tire_steering_angle,
        tire_steering_rate=current_state.dynamic_car_state.tire_steering_rate,
        time_point=current_state.time_point,
        vehicle_parameters=current_state.car_footprint.vehicle_parameters,
        is_in_auto_mode=current_state._is_in_auto_mode,
        angular_vel=current_state.dynamic_car_state.angular_velocity,
        angular_accel=current_state.dynamic_car_state.angular_acceleration,
    )
    states = [modified_current_state] + states[1:]
    return states

class AbstractMotionPlanner:
    @abstractmethod
    def compute_motion_trajectory(
        self,
        current_input: PlannerInput,
        target_centerline: List[StateSE2],
        centerline_offset: float,
        speed_limit_mps: float,
    ) -> AbstractTrajectory:
        pass

    @abstractmethod
    def initialize(self, initialization: PlannerInitialization) -> None:
        pass