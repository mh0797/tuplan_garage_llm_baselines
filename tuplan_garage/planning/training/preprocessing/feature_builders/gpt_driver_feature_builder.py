from typing import Dict, List, Tuple, Type

import numpy as np

from nuplan.common.actor_state.agent import Agent
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import build_ego_features
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from tuplan_garage.planning.training.preprocessing.features.gtp_driver_features import GPTDriverFeatures
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from tuplan_garage.planning.training.preprocessing.feature_builders.utils import CenterlineExtractor
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.route_utils import route_roadblock_correction
from tuplan_garage.planning.simulation.planner.llm_planner.utils import load_route_dicts

def get_corrected_route_lane_dict(
        map_api: AbstractMap,
        ego_state: EgoState,
        route_roadblock_ids: List[int],
    ) -> Tuple[
            Dict[str, LaneGraphEdgeMapObject], 
            Dict[str, RoadBlockGraphEdgeMapObject]
        ]:
        # Load initial roadblock_dict
        route_roadblock_dict, _ = load_route_dicts(
            route_roadblock_ids=route_roadblock_ids,
            map_api=map_api
        )
        # Find corrected route_roadblock_ids
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_dict
        )
        # Load updated roadblock_dict
        route_roadblock_dict, route_lane_dict = load_route_dicts(
            route_roadblock_ids=route_roadblock_ids,
            map_api=map_api
        )
        return route_lane_dict, route_roadblock_dict

class GPTFeatureBuilder(AbstractFeatureBuilder):
    """
    Builder for constructing GPT features during training and simulation.
    Features include
        - surrounding agents (position, orientation and speed)
        - ego state (speed)
        - relevant on-route centerlines 
    """

    def __init__(
        self,
        num_surrounding_agents: int,
        num_surrounding_objects: int,
        centerline_resolution: float,
        min_centerline_length: float,
        num_past_poses: int = 4,
        past_time_horizon: float = 2.0,
    ) -> None:
        self.num_surrounding_agents = num_surrounding_agents
        self.num_surrounding_objects = num_surrounding_objects
        self.centerline_resolution = centerline_resolution
        self.min_centerline_length = min_centerline_length
        self._roadblock_lookahead = 3

        self.centerline_extractor = CenterlineExtractor(
            resolution=centerline_resolution,
            max_length=min_centerline_length,
        )
        self.num_past_poses = num_past_poses
        self.past_time_horizon = past_time_horizon

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Type of the built feature."""
        return GPTDriverFeatures

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        return "gpt_features"

    def get_features_from_scenario(self, scenario: AbstractScenario) -> GPTDriverFeatures:
        """Inherited, see superclass."""
        current_ego_state = scenario.initial_ego_state
        detections = scenario.initial_tracked_objects
        map_api = scenario.map_api

        past_ego_states = scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        )
        sampled_past_ego_states = list(past_ego_states) + [current_ego_state]
        route_roadblock_ids = scenario.get_route_roadblock_ids()

        return self._compute_feature(
            ego_states=sampled_past_ego_states,
            detections=detections,
            map_api=map_api,
            route_roadblock_ids=route_roadblock_ids
        )

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> GPTDriverFeatures:
        """Inherited, see superclass."""
        history = current_input.history
        present_ego_state = history.ego_states[-1]
        past_ego_states = history.ego_states[:-1]
        observation = history.observations[-1]
        map_api = initialization.map_api
        route_roadblock_ids = initialization.route_roadblock_ids

        indices = sample_indices_with_time_horizon(
            self.num_past_poses, self.past_time_horizon, history.sample_interval
        )
        sampled_past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)]
        sampled_past_ego_states = sampled_past_ego_states + [present_ego_state]


        return self._compute_feature(
            ego_states=sampled_past_ego_states,
            detections=observation,
            map_api=map_api,
            route_roadblock_ids=route_roadblock_ids,
        )

    def _compute_feature(
        self,
        ego_states: List[EgoState],
        detections: DetectionsTracks,
        map_api: AbstractMap,
        route_roadblock_ids: List[int],
    ) -> GPTDriverFeatures:

        current_ego_state = ego_states[-1]
        route_lane_dict, route_roadblock_dict = get_corrected_route_lane_dict(
            map_api=map_api,
            ego_state=current_ego_state,
            route_roadblock_ids=route_roadblock_ids
        )

        drivable_area_map: PDMOccupancyMap = get_drivable_area_map(
            map_api=map_api,
            ego_state=current_ego_state,
        )

        centerlines = self.centerline_extractor.get_centerlines(
            ego_state=current_ego_state,
            route_lane_dict=route_lane_dict,
            route_roadblock_dict=route_roadblock_dict,
            drivable_area_map=drivable_area_map,
            only_on_route=True,
        )
        centerline_representation = {
            key:  (
                [
                    [
                        np.round(pose.x, 2),
                        np.round(pose.y, 2),
                        np.round(pose.heading, 2),
                    ]
                    for pose in lane_sequence
                ] 
                if lane_sequence else None
            )
            for key, lane_sequence in centerlines.items()
        }
        
        # extract ego state
        ego_dynamic_car_state = np.array([
            current_ego_state.dynamic_car_state.center_velocity_2d.x,
            current_ego_state.dynamic_car_state.center_velocity_2d.y,
            current_ego_state.dynamic_car_state.angular_velocity,
            current_ego_state.dynamic_car_state.center_acceleration_2d.x,
            current_ego_state.dynamic_car_state.center_acceleration_2d.y,
            current_ego_state.dynamic_car_state.angular_acceleration
        ])
        
        ego_poses = build_ego_features(ego_states, reverse=True)
        
        # extract agents representation
        agents = self._get_agents_representation(
            detections=detections,
            origin_absolute_state=current_ego_state.center,
            max_distance=50.0,
        )

        # extract objects representation
        objects = self._get_objects_representation(
            detections=detections,
            origin_absolute_state=current_ego_state.center,
            max_distance=50.0,
        )

        return GPTDriverFeatures(
            ego_state=[ego_dynamic_car_state],
            ego_poses=[ego_poses],
            centerlines=[centerline_representation],
            agents=[agents],
            objects=[objects],
        )
    
    def _get_objects_representation(
        self,
        detections: DetectionsTracks,
        origin_absolute_state: StateSE2,
        max_distance: float,
    ) -> List[np.array]:
        # extract surrounding objects
        tracked_objects: List[TrackedObject] = (
            detections.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.GENERIC_OBJECT)
            + detections.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.TRAFFIC_CONE)
        )
        local_object_poses = convert_absolute_to_relative_poses(
            origin_absolute_state=origin_absolute_state,
            absolute_states=[obj.center for obj in tracked_objects]
        )

        # filter for closest n agents
        distance_to_ego = [
            np.linalg.norm(pose[:2])
            for pose in local_object_poses
        ]
        closest_objects_indices = sorted(
            range(len(distance_to_ego)), 
            key=lambda k: distance_to_ego[k]
        )[:self.num_surrounding_objects]
        local_object_poses = [
            local_object_poses[k]
            for k in closest_objects_indices
            if distance_to_ego[k] < max_distance
        ]

        return [pose for pose in local_object_poses]

    def _get_agents_representation(
        self,
        detections: DetectionsTracks,
        origin_absolute_state: StateSE2,
        max_distance: float,
    ) -> List[np.array]:
        # extract surrounding agents
        tracked_vehicles: List[Agent] = detections.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
        agent_velocities = np.array([[agent.velocity.x, agent.velocity.y] for agent in tracked_vehicles])
        local_agent_poses = convert_absolute_to_relative_poses(
            origin_absolute_state=origin_absolute_state,
            absolute_states=[agent.center for agent in tracked_vehicles]
        )

        # filter for closest n agents
        distance_to_ego = [
            np.linalg.norm(pose[:2])
            for pose in local_agent_poses
        ]
        closest_agent_indices = sorted(
            range(len(distance_to_ego)), 
            key=lambda k: distance_to_ego[k]
        )[:self.num_surrounding_agents]

        local_agent_poses = [
            local_agent_poses[k]
            for k in closest_agent_indices
            if distance_to_ego[k] < max_distance
        ]
        agent_velocities = [
            agent_velocities[k] 
            for k in closest_agent_indices
            if distance_to_ego[k] < max_distance
        ]
        
        return [
            np.concatenate([agent_pose, agent_velocities], axis=-1)
            for agent_pose, agent_velocities
            in zip(local_agent_poses, agent_velocities)
        ]
