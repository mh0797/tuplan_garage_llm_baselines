from typing import Dict, List, Tuple, Type

import numpy as np

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from tuplan_garage.planning.training.preprocessing.features.llm_features import AvailableCenterlines, Centerline, LLMFeatures, SurroundingVehicles
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from tuplan_garage.planning.training.preprocessing.feature_builders.utils import CenterlineExtractor
from tuplan_garage.planning.training.preprocessing.feature_builders.gpt_driver_feature_builder import get_corrected_route_lane_dict
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from shapely.geometry.point import Point

class LLMFeatureBuilder(AbstractFeatureBuilder):
    """
    Builder for constructing LLM features during training and simulation.
    Features include
        - surrounding agents (position, orientation and speed)
        - ego state (speed)
        - relevant on-route centerlines 
    """

    def __init__(
        self,
        centerline_resolution: float,
        min_centerline_length: float,
        distance_behind: float,
    ) -> None:
        self.centerline_extractor = CenterlineExtractor(
            resolution=centerline_resolution,
            max_length=min_centerline_length,
            distance_behind=distance_behind,
        )

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Type of the built feature."""
        return LLMFeatures

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        return "gpt_features"

    def get_features_from_scenario(self, scenario: AbstractScenario) -> LLMFeatures:
        """Inherited, see superclass."""
        ego_state = scenario.initial_ego_state
        detections = scenario.initial_tracked_objects
        map_api = scenario.map_api

        route_roadblock_ids = scenario.get_route_roadblock_ids()

        return self._compute_feature(
            ego_state=ego_state,
            detections=detections,
            map_api=map_api,
            route_roadblock_ids=route_roadblock_ids
        )

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> LLMFeatures:
        """Inherited, see superclass."""
        history = current_input.history
        ego_state = history.ego_states[-1]
        observation = history.observations[-1]
        map_api = initialization.map_api
        route_roadblock_ids = initialization.route_roadblock_ids

        return self._compute_feature(
            ego_state=ego_state,
            detections=observation,
            map_api=map_api,
            route_roadblock_ids=route_roadblock_ids,
        )

    def _compute_feature(
        self,
        ego_state: EgoState,
        detections: DetectionsTracks,
        map_api: AbstractMap,
        route_roadblock_ids: List[int],
    ) -> LLMFeatures:

        route_lane_dict, route_roadblock_dict = get_corrected_route_lane_dict(
            map_api=map_api,
            ego_state=ego_state,
            route_roadblock_ids=route_roadblock_ids
        )

        drivable_area_map: PDMOccupancyMap = get_drivable_area_map(
            map_api=map_api,
            ego_state=ego_state,
        )

        centerlines = self.centerline_extractor.get_available_centerlines(
            ego_state=ego_state,
            route_lane_dict=route_lane_dict,
            route_roadblock_dict=route_roadblock_dict,
            drivable_area_map=drivable_area_map,
        )

        oncoming_lane = self.get_oncoming_lane(
            centerlines=centerlines,
            map_api=map_api,
            ego_state=ego_state,
        )
        
        # extract current speed
        current_speed = np.array([ego_state.dynamic_car_state.speed])
        
        # extract agents representation
        agents = self._get_agents_representation(
            detections=detections,
            origin_ego_state=ego_state,
            centerlines=centerlines,
            oncoming_lane=oncoming_lane,
        )

        traffic_cones = self._get_traffic_cones_representation(
            detections=detections,
            centerlines=centerlines,
            origin_ego_state=ego_state,
        )

        return LLMFeatures(
            ego_speed=[current_speed],
            centerlines=[centerlines],
            agents=[agents],
            traffic_cones=[traffic_cones],
        )

    def get_oncoming_lane(self, centerlines: AvailableCenterlines, map_api: AbstractMap, ego_state:EgoState) -> Centerline:
        if centerlines.left is not None or centerlines.right is not None:
            # we only look for oncoming lanes if no other lane is available
            return None
        else:
            # find candidates for oncoming lane on left side
            current_lane: MapObject = (
                map_api.get_map_object(centerlines.current.id, SemanticMapLayer.LANE)
                or
                map_api.get_map_object(centerlines.current.id, SemanticMapLayer.LANE_CONNECTOR)
            )
            # lane_heading = current_lane.baseline_path.discrete_path[0].heading
            candidate_proximal_lanes_left = map_api.get_proximal_map_objects(
                point=current_lane.left_boundary.discrete_path[0].point,
                radius=5.0,
                layers=[SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
            )
            candidate_oncoming_lanes_left = [
                lane for lanes in candidate_proximal_lanes_left.values() for lane in lanes
                if lane.left_boundary.id == current_lane.left_boundary.id
                and lane.id != current_lane.id
            ]
            # find candidates for oncoming lane on right side
            candidate_proximal_lanes_right = map_api.get_proximal_map_objects(
                point=current_lane.right_boundary.discrete_path[0].point,
                radius=5.0,
                layers=[SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
            )
            candidate_oncoming_lanes_right = [
                lane for lanes in candidate_proximal_lanes_right.values() for lane in lanes
                if lane.right_boundary.id == current_lane.right_boundary.id
                and lane.id != current_lane.id
            ]

            # find best candidate
            candidate_oncoming_lanes = candidate_oncoming_lanes_left + candidate_oncoming_lanes_right
            if len(candidate_oncoming_lanes) == 0 or len(candidate_oncoming_lanes) > 1:
                # cannot determine oncoming lane without doubt
                # TODO: select based on heuristic, e.g. ADE in case of len>1
                return None
            else:
                oncoming_lane = candidate_oncoming_lanes[0]
                path = oncoming_lane.baseline_path.discrete_path
                incoming_edges = oncoming_lane.incoming_edges
                num_lanes = 1
                # extend oncoming lane with incoming edge:
                while len(incoming_edges) == 1 and num_lanes < 5:
                    # TODO: if len(incoming_edges) > 1, select based on heuristic, e.g. ADE
                    path.extend(incoming_edges[0].baseline_path.discrete_path)
                    incoming_edges = incoming_edges[0].incoming_edges
                    num_lanes += 1
                
                lane_poses = convert_absolute_to_relative_poses(
                    origin_absolute_state=ego_state.center,
                    absolute_states=path,
                )
                lane_poses = [i for i in lane_poses]
                return Centerline(
                        id=oncoming_lane.id,
                        successor_id=None,
                        poses=lane_poses,
                        required_lane_changes_to_route=None,
                        speed_limit=oncoming_lane.speed_limit_mps,
                    )

    def _get_traffic_cones_representation(
        self,
        detections: DetectionsTracks,
        centerlines: AvailableCenterlines,
        origin_ego_state: EgoState,
    ) -> SurroundingVehicles:
        # extract surrounding agents
        tracked_cones: List[Agent] = detections.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.TRAFFIC_CONE)
        tracked_generic_objects: List[Agent] = detections.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.GENERIC_OBJECT)
        tracked_objects = tracked_cones + tracked_generic_objects

        centerline_paths = {
            k: PDMPath(
                [
                    StateSE2(x=pose[0], y=pose[1], heading=pose[2])
                    for pose in getattr(centerlines,k).poses
                ]
            )
            if getattr(centerlines,k)
            else None
            for k in ["left", "right", "current"]
        }

        # assign agents to lanes
        cone_lane_assignment = self._assign_tracked_object_to_lanes(
            agents=tracked_objects,
            centerline_paths=centerline_paths,
            origin_absolute_state=origin_ego_state.center,
            max_offset_from_centerline=0.5,
        )

        # filter agents to keep only one vehicle ahead and behind ego
        relevant_cones = self._filter_agents(
            agent_lane_assignment=cone_lane_assignment,
            centerline_paths=centerline_paths,
        )

        cones_annotations = {
            lane_role: [self._extract_agent_features(agent, origin_ego_state.center) if agent else None for agent in agents]
            for (lane_role,agents) in relevant_cones.items()
        }

        return SurroundingVehicles(
            front_right=cones_annotations["right"][1],
            rear_right=None,
            front_left=cones_annotations["left"][1],
            rear_left=None,
            front_current=cones_annotations["current"][1],
            rear_current=None,
            leading_stationary_agent=None,
            oncoming_agent=None,
        )
        

    def _assign_tracked_object_to_lanes(
        self,
        agents: List[TrackedObject],
        centerline_paths: Dict[str, PDMPath],
        origin_absolute_state: StateSE2,
        max_offset_from_centerline: float=5.0,
    ) -> Dict[str, List[Tuple[float, TrackedObject]]]:
        '''
        Assigns agents to the available lanes.
        Agents with an offset of more than max_offset_from_centreline meters to any centerline are excluded.
        :return Dict[str, [Tuple[float, Agent]]], where key is the lane role, and the first entry of value is the progress along the lane
        '''
        agent_centerline_assignment = {k: [] for k in centerline_paths.keys()}
        for agent in agents:
            # find lane with smallest offset
            agent_center = Point(
                convert_absolute_to_relative_poses(
                    origin_absolute_state=origin_absolute_state,
                    absolute_states=[agent.center]
                )[0,:2]
            )
            
            offset_to_lanes = {k: agent_center.distance(path.linestring) for (k,path) in centerline_paths.items() if path}
            closest_lane, offset = min(offset_to_lanes.items(), key=lambda x: x[1])
            if offset < max_offset_from_centerline:
                # calculate progress
                progress = centerline_paths[closest_lane].project(agent_center)
                # append agent to respective list
                agent_centerline_assignment[closest_lane].append((progress, agent))

        return agent_centerline_assignment

    def _filter_agents(
        self,
        agent_lane_assignment: Dict[str, List[Tuple[float, Agent]]],
        centerline_paths: Dict[str, PDMPath],
    ) -> Dict[str, List[Agent]]:

        filtered_agents = {}
        for lane_role in centerline_paths.keys():
            if centerline_paths[lane_role] is None:
                filtered_agents.update({lane_role: [None, None]})
            else:
                # local coordinates, so ego is located at 0,0
                ego_progress = centerline_paths[lane_role].project(Point(0,0))
                _, leading_vehicle = min(
                    [
                        agent for agent in
                        agent_lane_assignment[lane_role]
                        if agent[0]-ego_progress > 0
                    ],
                    key=lambda x: x[0]-ego_progress,
                    default=(0, None),
                )
                _, following_vehicle = max(
                    [
                        agent for agent in
                        agent_lane_assignment[lane_role]
                        if agent[0]-ego_progress < 0
                    ],
                    key=lambda x: x[0]-ego_progress,
                    default=(0, None),
                )
                filtered_agents.update({lane_role: [following_vehicle, leading_vehicle]})
        return filtered_agents

    def _extract_agent_features(
        self,
        agent: Agent,
        origin_absolute_state: StateSE2,
    ) -> np.ndarray:
        local_agent_pose = convert_absolute_to_relative_poses(
            origin_absolute_state=origin_absolute_state,
            absolute_states=[agent.center]
        )[0]
        agent_speed = np.array([agent.velocity.magnitude()])
        agent_features = np.concatenate([local_agent_pose, agent_speed], axis=-1)
        return agent_features

    def _get_agents_representation(
        self,
        detections: DetectionsTracks,
        origin_ego_state: EgoState,
        centerlines: AvailableCenterlines,
        oncoming_lane: Centerline,
    ) -> SurroundingVehicles:
        # TODO: This implementation is not able to include agents in previous roadblocks

        # extract surrounding agents
        tracked_vehicles: List[Agent] = detections.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
        centerline_paths = {
            k: PDMPath(
                [
                    StateSE2(x=pose[0], y=pose[1], heading=pose[2])
                    for pose in getattr(centerlines,k).poses
                ]
            )
            if getattr(centerlines,k)
            else None
            for k in ["left", "right", "current"]
        }
        if oncoming_lane is not None:
            centerline_paths["oncoming"] = PDMPath(
                [
                    StateSE2(x=pose[0], y=pose[1], heading=pose[2])
                    for pose in oncoming_lane.poses
                ]
            )

        # assign agents to lanes
        agent_lane_assignment = self._assign_tracked_object_to_lanes(
            agents=tracked_vehicles,
            centerline_paths=centerline_paths,
            origin_absolute_state=origin_ego_state.center,
            max_offset_from_centerline=0.5,
        )
        agent_lane_assignment = self._filter_parked_vehicles(agent_lane_assignment)

        # filter agents to keep only one vehicle ahead and behind ego
        relevant_agents = self._filter_agents(
            agent_lane_assignment=agent_lane_assignment,
            centerline_paths=centerline_paths,
        )

        # extract speeds and poses from agents
        agent_annotations = {
            lane_role: [self._extract_agent_features(agent, origin_ego_state.center) if agent else None for agent in agents]
            for (lane_role,agents) in relevant_agents.items()
        }

        # find stationary agents with a footprint that overlaps the current lane
        included_agents=[a.token for agents_on_lane in relevant_agents.values() for a in agents_on_lane if a]
        stationary_agents_on_current_lane = self._assign_tracked_object_to_lanes(
            agents=[agent for agent in tracked_vehicles if agent.velocity.magnitude() == 0.0 and agent.token not in included_agents],
            centerline_paths={"current": centerline_paths["current"]},
            origin_absolute_state=origin_ego_state.center,
            max_offset_from_centerline=2.5,
        )
        closest_stationary_agent = self._filter_agents(
            agent_lane_assignment=stationary_agents_on_current_lane,
            centerline_paths={"current": centerline_paths["current"]},
        )["current"][1]

        if closest_stationary_agent is not None:
            closest_stationary_agent_progress = [
                a for a in stationary_agents_on_current_lane["current"]
                if a[1].token == closest_stationary_agent.token
            ][0][0]
            stationary_agent_annotation = self._extract_notable_agent_features(
                agent=closest_stationary_agent,
                centerline=centerline_paths["current"],
                origin_ego_state=origin_ego_state,
                progress=closest_stationary_agent_progress,
            )
        else:
            stationary_agent_annotation = None

        if oncoming_lane is not None and len(agent_lane_assignment["oncoming"]) > 0:
            oncoming_agent_annotation = agent_annotations["oncoming"][0]
        else:
            oncoming_agent_annotation = None


        return SurroundingVehicles(
            front_right=agent_annotations["right"][1],
            rear_right=agent_annotations["right"][0],
            front_left=agent_annotations["left"][1],
            rear_left=agent_annotations["left"][0],
            front_current=agent_annotations["current"][1],
            rear_current=agent_annotations["current"][0],
            leading_stationary_agent=stationary_agent_annotation,
            oncoming_agent=oncoming_agent_annotation,
        )

    def _filter_parked_vehicles(self, agent_lane_assignment: Dict[str, List[Tuple[float, Agent]]]) -> Dict[str, List[Tuple[float, Agent]]]:
        return {
            lane: [a for a in agents if a[1].velocity.magnitude() > 0]
            for lane, agents
            in agent_lane_assignment.items()
        }

    def _extract_notable_agent_features(
        self,
        agent: Agent,
        progress: float,
        centerline: PDMPath,
        origin_ego_state: EgoState
    ):
        agent_features = self._extract_agent_features(agent, origin_ego_state.center)
        agent_center = convert_absolute_to_relative_poses(
                origin_absolute_state=origin_ego_state.center,
                absolute_states=[agent.center]
            )[0]
        agent_center = StateSE2(x=agent_center[0], y=agent_center[1], heading=agent_center[2])
        closest_point: StateSE2 = centerline.interpolate(distances=[progress])[0]
        agent_offset = agent_center.distance_to(closest_point)
        relative_heading = agent_center.heading - closest_point.heading
        effective_half_agent_width = max(
            abs(-agent.box.half_length*np.sin(relative_heading) + agent.box.half_width*np.cos(relative_heading)),
            abs(agent.box.half_length*np.sin(relative_heading) + agent.box.half_width*np.cos(relative_heading)),
        )
        required_offset = origin_ego_state.car_footprint.half_width + effective_half_agent_width - abs(agent_offset)
        # the the agent offset is large enough to acomodate for half width of ego and the agent, no offset is required
        required_offset = max(0, required_offset)
        # if vehicle is left of ego, we pass it to the right
        required_offset = np.array([required_offset * np.sign(-1.0 * agent_center.y)])
        # required_offset = np.array([1.27])
        return np.concatenate(
            [agent_features, required_offset],
            axis=-1
        )