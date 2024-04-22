from abc import ABC
from functools import cached_property
import logging
import numpy as np
from typing import Tuple, Union, List
import numpy.typing as npt

from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
from tuplan_garage.planning.training.preprocessing.features.llm_features import LLMFeatures
from tuplan_garage.planning.simulation.planner.llm_planner.behavior_planner.abstract_llm_behavior_planner import LaneFollowingBehavior
from tuplan_garage.planning.simulation.planner.llm_planner.llm_interface.llm_interface import AbstractLLMInterface
from tuplan_garage.planning.training.preprocessing.features.llm_features import AvailableCenterlines, Centerline

logger = logging.getLogger(__name__)

class LLMDriverBehaviorPlanner(ABC):
    def __init__(
            self,
            llm_interface: AbstractLLMInterface,
            use_examples: bool,
    ):
        self.llm_interface = llm_interface
        self.use_examples = use_examples
    
    def get_comment_on_current_behavior(self) -> str:
        return self._current_response

    def get_prompt_for_current_behavior(self) -> str:
        return self.system_prompt + "\n\n" + self._current_prompt

    @cached_property
    def in_context_examples(self) -> List[Tuple[str,str]]:
        if self.use_examples:
            return [
                (
                    (
                        'Perception:\n'
                        ' - Vehicle: in the current lane at 20.5 meters ahead driving 0.8 meters per '
                        'second faster than you.\n'
                        ' - Oncoming Traffic: You are in a road with multiple lanes and no oncoming '
                        'traffic. Use #merge_left# or #merge_right# to overtake.\n'
                        ' - Vehicle: in the left lane at 33.3 meters ahead driving 2.0 meters per '
                        'second slower than you.\n'
                        ' - Vehicle: in the right lane at 51.1 meters ahead driving 1.1 meters per '
                        'second slower than you.\n'
                        'Availables lanes:\n'
                        ' - current lane: the lane that you are currently driving in, you can follow '
                        'it to reach your destination\n'
                        ' - left lane: available and on-route\n'
                        ' - right lane: available and on-route\n'
                        'Ego States:\n'
                        ' - Velocity: 7.9 meters per second\n'
                        ' - Speed Limit: 13.4  meters per second\n'
                        'Mission Goal: Decide to #follow_current# or #merge_left# or #merge_right#'
                    ),(
                        '- Thoughts:\n'
                        '  - Notable Objects\n'
                        '    - Leading vehicle in the current lane is faster. We can accelerate to '
                        'its speed.\n'
                        '    - Leading vehicle in the left lane is slower than us. Merging to the '
                        'left lane requires slowing down.\n'
                        '    - There is no vehicle behind in the left lane, thus, merging safely is '
                        'possible.\n'
                        '    - Leading vehicle in the right lane is slower than us. Merging to the '
                        'right lane requires slowing down.\n'
                        '    - There is no vehicle behind in the right lane, thus, merging safely is '
                        'possible.\n'
                        '\n'
                        '  - Potential Effects\n'
                        '    - All lanes are on route. However, the current lane is the fastest. '
                        'Hence, we need to follow it.\n'
                        '\n'
                        '- Behavior Planning\n'
                        'appropriate decision #follow_current#'
                    )
                ),
                (
                    (
                        'Perception:\n'
                        ' - Vehicle: in the current lane at 38.4 meters ahead driving 0.1 meters per '
                        'second slower than you.\n'
                        ' - Oncoming Traffic: You are in a road with multiple lanes and no oncoming '
                        'traffic. Use #merge_left# or #merge_right# to overtake.\n'
                        ' - Vehicle: in the left lane at 0.4 meters ahead driving 0.2 meters per '
                        'second faster than you.\n'
                        ' - Vehicle: in the right lane at 30.3 meters ahead driving 0.0 meters per '
                        'second slower than you.\n'
                        ' - Vehicle: in the left lane at 4.9 meters ahead driving 2.3 meters per '
                        'second slower than you.\n'
                        'Availables lanes:\n'
                        ' - current lane: the lane that you are currently driving in, you need to '
                        'merge to another lane to follow the route\n'
                        ' - left lane: available, but not on-route\n'
                        ' - right lane: available, but not on-route\n'
                        'Ego States:\n'
                        ' - Velocity: 12.8 meters per second\n'
                        ' - Speed Limit: 13.4  meters per second\n'
                        'Mission Goal: Decide to #follow_current# or #merge_left#'
                    ),(
                        '- Thoughts:\n'
                        '  - Notable Objects\n'
                        '    - Leading vehicle in the current lane is slower. Overtaking it might '
                        'enable making more progress.\n'
                        '    - Leading vehicle in the left lane is faster. Hence, merging to the '
                        'left lane may allow more progress.\n'
                        '    - Vehicle behind in the left lane is slower but close. Merging safely '
                        'is not possible. We need to wait for the gap to increase or accelerate.\n'
                        '\n'
                        '  - Potential Effects\n'
                        '    - We need to merge to the left lane to get to the destination and '
                        'merging can be done safely. Hence, we should merge to the left lane.\n'
                        '\n'
                        '- Behavior Planning\n'
                        'appropriate decision #merge_left#'
                    )
                ),
                (
                    (
                        'Perception:\n'
                        ' - Vehicle: in the current lane at 5.9 meters ahead driving 2.0 meters per '
                        'second faster than you.\n'
                        ' - Oncoming Traffic: You are in a road with multiple lanes and no oncoming '
                        'traffic. Use #merge_left# or #merge_right# to overtake.\n'
                        ' - Vehicle: in the left lane at 8.0 meters ahead driving 2.8 meters per '
                        'second faster than you.\n'
                        ' - Vehicle: in the left lane at 0.9 meters behind driving 0.3 meters per '
                        'second faster than you.\n'
                        ' - Vehicle: in the right lane at 5.4 meters behind driving 3.1 meters per '
                        'second faster than you.\n'
                        'Availables lanes:\n'
                        ' - current lane: the lane that you are currently driving in, you can follow '
                        'it to reach your destination\n'
                        ' - left lane: available and on-route\n'
                        ' - right lane: available and on-route\n'
                        'Ego States:\n'
                        ' - Velocity: 7.5 meters per second\n'
                        ' - Speed Limit: 13.4  meters per second\n'
                        'Mission Goal: Decide to #follow_current# or #merge_left# or #merge_right#'
                    ),(
                        '- Thoughts:\n'
                        '  - Notable Objects\n'
                        '    - Leading vehicle in the current lane is faster. We can accelerate to '
                        'its speed.\n'
                        '    - Leading vehicle in the left lane is faster. Hence, merging to the '
                        'left lane may allow more progress.\n'
                        '    - Vehicle behind in the left lane is faster but still far enough away. '
                        'Merging before it safely is possible.\n'
                        '    - There is no leading vehicle in the right lane. We could drive there '
                        'at speed limit.\n'
                        '    - Vehicle behind in the right lane is faster but still far enough away. '
                        'Merging before it safely is possible.\n'
                        '\n'
                        '  - Potential Effects\n'
                        '    - The right lane would allow the fastest progress. Merging would be '
                        'safe so we should merge to and follow the right lane.\n'
                        '\n'
                        '- Behavior Planning\n'
                        'appropriate decision #merge_right#'
                    )
                ),
                (
                    (
                        'Perception:\n'
                        ' - Traffic Cone: at 5.8 meters ahead. Your current lane is blocked.\n'
                        ' - Oncoming Traffic: in the oncoming lane at 65.0 meters ahead approaching '
                        'at -1.3 meters per second.\n'
                        'Availables lanes:\n'
                        ' - current lane: the lane that you are currently driving in, you can follow '
                        'it to reach your destination\n'
                        ' - left lane: not available\n'
                        ' - right lane: not available\n'
                        'Ego States:\n'
                        ' - Velocity: 6.0 meters per second\n'
                        ' - Speed Limit: 13.4  meters per second\n'
                        'Mission Goal: Decide to #overtake_obstacle# or #stop_and_wait#'
                    ),(
                        '- Thoughts:\n'
                        '  - Notable Objects\n'
                        '    - There is a traffic cone in the current lane. We need to merge to a '
                        'different lane or drive around it.\n'
                        '    - The vehicle in the opposite lane is still far away, so we can '
                        'overtake safely.\n'
                        '\n'
                        '  - Potential Effects\n'
                        '    - The current lane is blocked. Oncoming traffic is still far away and '
                        'slow enough so that we can safely overtake the obstacle without risking a '
                        'collision.\n'
                        '\n'
                        '- Behavior Planning\n'
                        'appropriate decision #overtake_obstacle#'
                    ),
                ),
                (
                    (
                        'Perception:\n'
                        ' - Traffic Cone: at 6.3 meters ahead. Your current lane is blocked.\n'
                        ' - Oncoming Traffic: in the oncoming lane at 87.2 meters ahead approaching '
                        'at -7.8 meters per second.\n'
                        'Availables lanes:\n'
                        ' - current lane: the lane that you are currently driving in, you can follow '
                        'it to reach your destination\n'
                        ' - left lane: not available\n'
                        ' - right lane: not available\n'
                        'Ego States:\n'
                        ' - Velocity: 11.8 meters per second\n'
                        ' - Speed Limit: 13.4  meters per second\n'
                        'Mission Goal: Decide to #overtake_obstacle# or #stop_and_wait#'
                    ),(
                        '- Thoughts:\n'
                        '  - Notable Objects\n'
                        '    - There is a stationary vehicle in the current lane. It might be parked '
                        'or crashed. We need to merge to a different lane or drive around it.\n'
                        '    - There is an oncoming vehicle in the opposite lane. Thus, we need to '
                        'wait before overtaking safely.\n'
                        '\n'
                        '  - Potential Effects\n'
                        '    - The current lane is blocked. However, we need to wait for oncoming '
                        'traffic to pass before we can overtake it. Thus we need to stop and wait.\n'
                        '\n'
                        '- Behavior Planning\n'
                        'appropriate decision #stop_and_wait#'
                    )
                )
            ]
        else:
            return []

    @cached_property
    def system_prompt(self) -> str:
        return (
            "Autonomous Driving Planner\n"
            "Role: You are the brain of an autonomous vehicle. Chose a safe behavior from the set of mission goals. Avoid collisions with other objects.\n"
            "\n"
            "Context\n"
            "- Lanes: You can decide to follow any of the available lanes.\n"
            "- Route: Your overall goal is to reach your destination defined by the route. Not all available lanes are on route.\n"
            "- Objective: Decide on a safe driving behavior that allows you to make progress along the route.\n"
            "\n"
            "Inputs\n"
            "1. Perception: Info about surrounding objects.\n"
            "2. Availables lanes: The road lanes that you can reach from your current state.\n"
            "3. Ego-States: Your current state including velocity and speed limit.\n"
            "4. Mission Goal: a set of behaviors among which you need to decide.\n"
            "\n"
            "Task\n"
            "- Thought Process: Note down critical objects and potential effects from your perceptions and predictions.\n"
            "- Behavior Planning: Decide on a safe behavior which allows making progress along the route.\n"
            "\n"
            "Output\n"
            "- Thoughts:\n"
            " - Notable Objects\n"
            "   Potential Effects\n"
            "- Appropriate Behavior (MOST IMPORTANT): Decide for one of the options mentioned in 'Mission Goal'. "
            "It must be enclosed by the special character #, e.g., #follow_current#\n"
        )

    def _build_task_prompt(self, features: LLMFeatures, decimals: int=1) -> str:
        def _get_vehicle_description(vehicle_state_array: np.ndarray, ego_speed: float, lane:str, decimals:int=decimals) -> str:
            if vehicle_state_array is None:
                return ""
            else:
                ahead_behind = "ahead" if vehicle_state_array[0] > 0 else "behind"
                long_dist = np.round(np.abs(vehicle_state_array[0]), decimals=decimals)
                relative_speed = np.round(vehicle_state_array[3] - ego_speed, decimals)
                if relative_speed > 0:
                    faster_slower = "faster"
                else:
                    faster_slower = "slower"
                relative_speed = np.abs(relative_speed)
                return f"Vehicle: in the {lane} lane at {long_dist} meters {ahead_behind} driving {relative_speed} meters per second {faster_slower} than you."

        def _get_mission_goal(centerlines: AvailableCenterlines, current_lane_is_blocked:bool) -> str:
            available_lanes = [
                role
                for role in ["left","current", "right"]
                if getattr(centerlines, role) is not None
            ]
            required_lane_changes = {
                role: getattr(centerlines, role).required_lane_changes_to_route
                for role in available_lanes
            }
            on_route_lanes = [k for k,v in required_lane_changes.items() if v==0]
            closest_lane = min(required_lane_changes, key=lambda k: required_lane_changes[k])

            available_behaviors = [
                "follow_current",
                "overtake_obstacle",
                "merge_left",
                "merge_right",
                "stop_and_wait"
            ]

            if (
                centerlines.left is None
                or (
                    not current_lane_is_blocked
                    and ("left" not in on_route_lanes)
                    and (closest_lane != "left")
                )
            ):
                available_behaviors.remove("merge_left")
            
            if (
                centerlines.right is None
                or (
                    not current_lane_is_blocked
                    and ("right" not in on_route_lanes)
                    and (closest_lane != "right")
                )
            ):
                available_behaviors.remove("merge_right")

            if current_lane_is_blocked:
                available_behaviors.remove("follow_current")
            
            if (
                centerlines.left is not None
                or centerlines.right is not None
                or not current_lane_is_blocked
            ):
                available_behaviors.remove("overtake_obstacle")
                available_behaviors.remove("stop_and_wait")
            
            available_behaviors = ["#"+option+"#" for option in available_behaviors]
            return "Decide to " + " or ".join(available_behaviors)
        
        def _get_lane_available_prompt(centerline: Centerline) -> str:
            if centerline is None:
                return "not available"
            elif centerline.required_lane_changes_to_route > 0:
                return "available, but not on-route"
            else:
                return f"available and on-route"

        vehicles = features.agents[0]
        lines = features.centerlines[0]
        traffic_cones = features.traffic_cones[0]
        ego_speed = np.round(features.ego_speed[0], decimals=decimals)[0]
        speed_limit = np.round(lines.current.speed_limit, decimals=decimals)

        current_lane_is_blocked = False
        if traffic_cones.front_current is not None:
            long_dist = np.round(np.abs(traffic_cones.front_current[0]), decimals=decimals)
            traffic_cone_prompt = f"Traffic Cone: at {long_dist} meters ahead. Your current lane is blocked."
            current_lane_is_blocked = True
        else:
            traffic_cone_prompt = ""

        if vehicles.leading_stationary_agent is not None:
            long_dist = np.round(np.abs(vehicles.leading_stationary_agent[0]), decimals=decimals)
            leading_vehicle_prompt = f"Stationary Vehicle: at {long_dist} meters ahead. Your current lane is blocked."
            current_lane_is_blocked = True
        else:
            leading_vehicle_prompt = f"{_get_vehicle_description(vehicles.front_current, ego_speed, 'current')}"

        
        if vehicles.oncoming_agent is not None:
            long_dist = np.round(np.abs(vehicles.oncoming_agent[0]), decimals=decimals)
            speed = np.round(vehicles.oncoming_agent[3] - ego_speed, decimals)
            oncoming_agent_prompt = f"Oncoming Traffic: in the oncoming lane at {long_dist} meters ahead approaching at {speed} meters per second."
        else:
            if lines.left is not None or lines.right is not None:
                oncoming_agent_prompt = "Oncoming Traffic: You are in a road with multiple lanes and no oncoming traffic. Use #merge_left# or #merge_right# to overtake."
            else:
                oncoming_agent_prompt = "Oncoming Traffic: There is no oncoming traffic. You can safely overtake"

        
        perception_prompt = "Perception:\n"
        if traffic_cone_prompt != "":
            perception_prompt += f" - {traffic_cone_prompt}\n"
        if leading_vehicle_prompt != "":
            perception_prompt += f" - {leading_vehicle_prompt}\n"
        if oncoming_agent_prompt != "":
            perception_prompt += f" - {oncoming_agent_prompt}\n"
        if _get_vehicle_description(vehicles.front_left, ego_speed, 'left') != "":
            perception_prompt += f" - {_get_vehicle_description(vehicles.front_left, ego_speed, 'left')}\n"
        if _get_vehicle_description(vehicles.front_right, ego_speed, 'right') != "":
            perception_prompt += f" - {_get_vehicle_description(vehicles.front_right, ego_speed, 'right')}\n"
        if _get_vehicle_description(vehicles.rear_left, ego_speed, 'left') != "":
            perception_prompt += f" - {_get_vehicle_description(vehicles.rear_left, ego_speed, 'left')}\n"
        if _get_vehicle_description(vehicles.rear_right, ego_speed, 'right') != "":
            perception_prompt += f" - {_get_vehicle_description(vehicles.rear_right, ego_speed, 'right')}\n"

        if lines.current.required_lane_changes_to_route == 0:
            current_lane_on_route = "you can follow it to reach your destination"
        else:
            current_lane_on_route = "you need to merge to another lane to follow the route"

        task_prompt =  (
            f"{perception_prompt}"
            "Availables lanes:\n"
            f" - current lane: the lane that you are currently driving in, {current_lane_on_route}\n"
            f" - left lane: {_get_lane_available_prompt(lines.left)}\n"
            f" - right lane: {_get_lane_available_prompt(lines.right)}\n"
            "Ego States:\n"
            f" - Velocity: {ego_speed} meters per second\n"
            f" - Speed Limit: {speed_limit}  meters per second\n"
            f"Mission Goal: {_get_mission_goal(lines, current_lane_is_blocked)}"
        )

        return task_prompt
    
    def parse_behavior(self, selected_behavior, features: LLMFeatures) -> Tuple[LaneFollowingBehavior, int]:
        def _get_target_speed(speed_limit: float, leading_agent: Union[npt.NDArray, None]) -> float:
            if leading_agent is not None:
                return min(speed_limit, leading_agent[3])
            else:
                return speed_limit
        if selected_behavior == "follow_current":
            target_speed = _get_target_speed(
                speed_limit=features.centerlines[0].current.speed_limit,
                leading_agent=features.agents[0].front_current
            )
            target_lane_ids = [
                features.centerlines[0].current.id,
                features.centerlines[0].current.successor_id
            ]
            return LaneFollowingBehavior(
                centerline=features.centerlines[0].current.poses,
                centerline_offset=0.0,
                speed_limit=target_speed,
            ), target_lane_ids
        elif selected_behavior == "overtake_obstacle":
            leading_stationary_agent = features.agents[0].leading_stationary_agent
            if leading_stationary_agent is None:
                logger.warning("No stationary agent in front of ego, using follow_current as fallback")
                return self.parse_behavior("follow_current", features)
            else:
                required_offset = leading_stationary_agent[-1]
                target_lane_ids = [
                    features.centerlines[0].current.id,
                    features.centerlines[0].current.successor_id
                ]
                return LaneFollowingBehavior(
                    centerline=features.centerlines[0].current.poses,
                    centerline_offset=required_offset,
                    speed_limit=features.centerlines[0].current.speed_limit,
                ), target_lane_ids
        elif selected_behavior == "merge_left":
            if features.centerlines[0].left is None:
                logger.warning("Left lane is not available, using follow_current as fallback")
                return self.parse_behavior("follow_current", features)
            else:
                target_speed = _get_target_speed(
                    speed_limit=features.centerlines[0].current.speed_limit,
                    leading_agent=features.agents[0].front_left
                )
                target_lane_ids = [
                    features.centerlines[0].left.id,
                    features.centerlines[0].left.successor_id
                ]
                return LaneFollowingBehavior(
                    centerline=features.centerlines[0].left.poses,
                    centerline_offset=0.0,
                    speed_limit=features.centerlines[0].left.speed_limit,
                ), target_lane_ids
        elif selected_behavior == "merge_right":
            if features.centerlines[0].right is None:
                logger.warning("Right lane is not available, using follow_current as fallback")
                return self.parse_behavior("follow_current", features)
            else:
                target_speed = _get_target_speed(
                    speed_limit=features.centerlines[0].current.speed_limit,
                    leading_agent=features.agents[0].front_right
                )
                target_lane_ids = [
                    features.centerlines[0].right.id,
                    features.centerlines[0].right.successor_id
                ]
                return LaneFollowingBehavior(
                    centerline=features.centerlines[0].right.poses,
                    centerline_offset=0.0,
                    speed_limit=features.centerlines[0].right.speed_limit,
                ), target_lane_ids
        elif selected_behavior == "stop_and_wait":
            target_lane_ids = [
                features.centerlines[0].current.id,
                features.centerlines[0].current.successor_id
            ]
            return LaneFollowingBehavior(
                centerline=features.centerlines[0].current.poses,
                centerline_offset=0.0,
                speed_limit=0.0,
            ), target_lane_ids
        else:
            logger.warning(f"Unknown behavior {selected_behavior}, using follow_current as fallback")
            return self.parse_behavior("follow_current", features)

    def infer_behavior(self, features: LLMFeatures) -> LaneFollowingBehavior:
        task_prompt = self._build_task_prompt(features)
        self._current_prompt = task_prompt
        response = self.llm_interface.infer_model(
            system_prompt=self.system_prompt,
            task_prompt=task_prompt,
            in_context_examples=self.in_context_examples,
        )

        try:
            selected_behavior = response.split("#")[1::2][-1]
        except IndexError:
            logger.warning(f"Failed to parse behavior from response {response}, using follow_current as fallback")
            selected_behavior = "follow_current"
        behavior, target_lane_ids  = self.parse_behavior(selected_behavior, features)
        self._current_target_lane_ids = target_lane_ids
        self._selected_behavior = selected_behavior

        behavior_response = "PARSED BEHAVIOR\n" + selected_behavior + "\n" + "offsets: " + str(behavior.centerline_offset)
        self._current_response = response + "\n\n" + behavior_response

        return behavior

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._current_target_lane_ids = [None]
        self._selected_behavior = None
        self._current_response = "Not set"
        self._current_prompt = "Not set"
        self.llm_interface.initialize()