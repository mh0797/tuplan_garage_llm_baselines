from typing import List, Optional, Tuple

import numpy as np
from shapely import Point

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.pdm_proposal import (
    PDMProposalManager,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import parallel_discrete_path
from tuplan_garage.planning.simulation.planner.llm_planner.motion_planner.abstract_motion_planner import AbstractMotionPlanner

class PDMPPPlanner(PDMClosedPlanner, AbstractMotionPlanner):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        map_radius: float,
    ):
        super().__init__(
            idm_policies=idm_policies,
            lateral_offsets=lateral_offsets,
            trajectory_sampling=trajectory_sampling,
            proposal_sampling=proposal_sampling,
            map_radius=map_radius,
        )

    def compute_motion_trajectory(
        self,
        current_input: PlannerInput,
        target_centerline: List[StateSE2],
        centerline_offset: float,
        speed_limit_mps: float,
    ) -> AbstractTrajectory:
        centerline = parallel_discrete_path(target_centerline, centerline_offset)
        self._centerline = PDMPath(centerline)
        self._speed_limit_mps = speed_limit_mps
        return self.compute_planner_trajectory(
            current_input=current_input
        )

    def _update_proposal_manager(self, ego_state: EgoState):
        proposal_paths: List[PDMPath] = generate_lateral_profiles(
            centerline=self._centerline,
            current_state=ego_state,
            lateral_offsets=self._lateral_offsets,
        )

        self._proposal_manager = PDMProposalManager(
            lateral_proposals=proposal_paths,
            longitudinal_policies=self._idm_policies,
        )
        self._proposal_manager.update(self._speed_limit_mps)

def generate_lateral_profiles(
    centerline: PDMPath,
    current_state: EgoState,
    lat_profile_len: float=100.0,
    lat_profile_resolution: float=0.1,
    lateral_offsets: List[float]=None,
) -> List[PDMPath]:

    # project initial state to centerline
    current_progress = centerline.project(
        Point(
            current_state.rear_axle.x,
            current_state.rear_axle.y
        )
    )
    sampling_progress = np.arange(
        start=current_progress,
        stop=min(current_progress+lat_profile_len, centerline.length),
        step=lat_profile_resolution,
    )
    assert sampling_progress.shape[0] > 1, ("""
        Centerline cannot be shorter than the proposal.
        Consider increasing the centerline length in the feature-builder or decreasing the length of the proposal
    """)

    # perpendicular points on centerline (same for all proposals)
    centerline_discrete_path: List[StateSE2] = centerline.interpolate(
        distances=[
            p for p in sampling_progress
        ]
    ).tolist()

    # add additional paths with lateral offset of centerline
    if lateral_offsets is None:
        return [PDMPath(centerline_discrete_path)]
    else:
        lateral_profiles = [
            PDMPath(
                parallel_discrete_path(
                    discrete_path=centerline_discrete_path, offset=lateral_offset
                )
            )
            for lateral_offset in lateral_offsets
        ]
        return lateral_profiles