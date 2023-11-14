from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from torch import Tensor
import numpy as np
import logging
import json

from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
from tuplan_garage.planning.training.preprocessing.features.llm_features import LLMFeatures

from tuplan_garage.planning.training.preprocessing.features.llm_features import AvailableCenterlines, Centerline

logger = logging.getLogger(__name__)

def parse_response_to_dict(response: str, features: LLMFeatures) -> Dict[str, Any]:
    try:
        res = "{" + response.split("{")[1].split("}")[0] + "}" 
        res = res.replace("'","\"")
        res = json.loads(res)
    except: 
        logger.warning(f"Couldn't parse response \n {response}\n Using current centerline as fallback")
        res = {
            "lane": "current",
            "lateral_offsets": 0,
            "target_speed": features.centerlines[0].current.speed_limit
        }
    
    if not "lane" in res:
        logger.warning(f"response \n {response}\n does not contain a centerline. Using current centerline as fallback")
        res["lane"] = "current"

    if not "lateral_offsets" in res:
        logger.warning(f"response \n {response}\n does not contain lateral_offsets. Using [0] centerline as fallback")
        res["lateral_offsets"] = 0

    if not "target_speed" in res:
        logger.warning(f"response \n {response}\n does not contain target_speed. Using current speed limit as fallback")
        res["target_speed"] = features.centerlines[0].current.speed_limit
    
    try:
        cl = getattr(features.centerlines[0],res["lane"])
        assert cl is not None
    except:
        logger.warning(f"Selected centerline {res['lane']} is not available. Using current centerline as fallback")
        res["lane"] = "current"
    
    if not isinstance(res["lateral_offsets"], float):
        res["lateral_offsets"] = 0.0
    
    if not isinstance(res["target_speed"], (int, float)) or res["target_speed"] > getattr(features.centerlines[0],res["lane"]).speed_limit:
        res["target_speed"] = getattr(features.centerlines[0],res["lane"]).speed_limit

    res["lane"] = getattr(features.centerlines[0],res["lane"]).poses

    return res

@dataclass
class LaneFollowingBehavior:
    centerline: List[List[Tensor]]
    centerline_offset: List[float]
    speed_limit: float

class AbstractLLMBehaviorPlanner(ABC):
    def __init__(
        self,
        use_examples: bool,
        system_promt_start_token: str,
        system_promt_end_token: str,
        instruction_start_token: str,
        instruction_end_token: str,
        eos_token: str,
        bos_token: str,
    ):
        self._use_examples = use_examples
        self._system_promt_start_token = system_promt_start_token
        self._system_promt_end_token = system_promt_end_token
        self._instruction_start_token = instruction_start_token
        self._instruction_end_token = instruction_end_token
        self._eos_token = eos_token
        self._bos_token = bos_token

    @abstractmethod
    def infer_behavior(self, features: LLMFeatures) -> LaneFollowingBehavior:
        pass

    @abstractmethod
    def initialize(self, initialization: PlannerInitialization) -> None:
        pass

    @abstractmethod
    def get_comment_on_current_behavior(self) -> str:
        pass
    
    @abstractmethod
    def get_prompt_for_current_behavior(self) -> str:
        pass

    def _get_system_prompt(self) -> str:
        return (
            "You are a car driving expert. Your job is to decide on an appropriate behavior and give the causes for your decision.\n"
            "You get a set of options to chose from. Each option is given by a lane, which is either the current lane or the one to the left or right.\n"
            "Additionally, you should decide on a target speed along the lane and a desired offset from its centerline.\n"
            "You will get information on the vehicles that are currently driving on each lane.\n"
            "If the vehicle ahead is too slow, you can decide to change to another lane to overtake it.\n"
            "Your goal is to follow the route. If possible, you should change to a lane that follows the route.\n"
            "However, you should only decide to change lanes if it's safe to do so.\n"
            "If it is unsafe, you can decide to slow down, by setting a low value for 'target_speed' or to nudge slightly towards that lane by setting a 'lateral_offset'.\n"
            "Otherwise you can set the lateral_offsets to '[0]' and the target_speed to the value of the current speed limit.\n"
            "Give your answer in the following json format:\n"
            "start of json\n"
            "{\n"
            "\t'reason': 'There is only one lane',\n"
            "\t'lateral_offsets': [-1,0,1],\n"
            "\t'target_speed': 14.0,\n"
            "\t'lane': 'current'\n"
            "}\n"
            "end of json\n"
            "There are no other lanes than the ones mentioned in the options. If no 'left' lane is mentioned, then 'left' is not a valid answer.\n"
            "The Json output shoud start with 'start of json' and end with 'end of json'\n"
        )


    def _get_examples(self) -> List[Tuple[str, str]]:
        if self._use_examples:
            return [
                (
                    (
                    "You are driving at 11.9 meters per second on the current lane. The speed limit is 13.4 meters per second. You have the following options:\n"
                    "Proceed at the current lane and follow vehicle at 64.3 meters ahead driving at 9.4 meters per second.\n"
                    "Proceed on the left lane and merge between vehicle at 67.9 meters ahead driving at 9.8 meters per second and vehicle at 7.2 meters behind driving at 14.4 meters per second.\n"
                    "It is not safe to merge to this lane.\n"
                    "What is the correct behaviour?"
                    ),(
                    "{\n"
                    "\t'reason': 'The vehicle ahead is still far away and it is not safe to merge to the left lane because the vehicle behind is approaching fast.,'\n"
                    "\t'lane': 'current',\n"
                    "\t'lateral_offsets': [0],\n"
                    "\t'target_speed': 13.4\n"
                    "}\n"
                    )
                )
            ]
        else:
            return ""

    def _build_prompt(self, features: LLMFeatures, decimals: int=1) -> str:
        def _get_vehicle_description(vehicle_state_array: np.ndarray, ego_speed: float, decimals:int=decimals) -> str:
            ahead_behind = "ahead" if vehicle_state_array[0] > 0 else "behind"
            long_dist = np.round(np.abs(vehicle_state_array[0]), decimals=decimals)
            relative_speed = np.round(vehicle_state_array[3] - ego_speed, decimals)
            if relative_speed > 0:
                faster_slower = "faster"
            else:
                faster_slower = "slower"
            relative_speed = np.abs(relative_speed)
            return f"vehicle at {long_dist} meters {ahead_behind} driving {relative_speed} meters per second {faster_slower} than you"
        
        def _get_centerline_description(centerline: Centerline, role: str) -> str:
            lane_prompt = f"Proceed at the {role} lane"
            return lane_prompt

        def _get_route_description(centerlines: AvailableCenterlines) -> str:
            lane_changes = {
                role: getattr(centerlines, role).required_lane_changes_to_route
                for role in ["left","current", "right"]
                if getattr(centerlines, role) is not None
            }
            if all([n==0 for n in lane_changes.values()]):
                return "All lanes are on route.\n"
            elif any([n==0 for n in lane_changes.values()]):
                on_route_lanes = [k for k,v in lane_changes.items() if v==0]
                return "Only the following lanes are on route: " + ", ".join(on_route_lanes) + ".\n"
            else:
                closest_lane = min(lane_changes, key=lambda k: lane_changes[k])
                return f"The {closest_lane} lane is closest to the route.\n"
        
        def _get_safety_evaluation(
            leading_vehicle: np.ndarray,
            following_vehicle: np.ndarray,
            ego_speed: float,
            min_time_gap_s: float=1.0,
            follower_decel_max: float=4.0,
            ) -> str:
            vehicle_length = 6.0

            if leading_vehicle is not None:
                # length of vehicle
                leader_rel_speed = leading_vehicle[3]
                gap_to_leading_vehicle = max(leading_vehicle[0] - vehicle_length, 0)
                if leader_rel_speed < 0:
                    # if ego is faster, need to make sure there is enough space to break
                    safety_margin_ahead = (3*leader_rel_speed**2) / (2*gap_to_leading_vehicle) < follower_decel_max
                else:
                    # if the other vehicle is faster, a lane-change is ok 
                    # if the gap after the leading vehicle breaks to ego-speed is still > min_time_gap s
                    leader_decel_max = follower_decel_max
                    gap_at_same_speed = leader_rel_speed**2 / (2*leader_decel_max) + gap_to_leading_vehicle
                    safety_margin_ahead = gap_at_same_speed / ego_speed > min_time_gap_s
            else:
                safety_margin_ahead = True

            # calculate safety margin behind. TTC > min_ttc_s AND breaking < 5m/s^2
            if following_vehicle is not None:
                follower_distance = max(np.abs(following_vehicle[0]) - vehicle_length, 0)
                follower_rel_speed = following_vehicle[3]
                initial_time_gap = follower_distance / (follower_rel_speed+ego_speed)
                if initial_time_gap >= min_time_gap_s:
                    if follower_rel_speed > 0:
                        # following vehicle is faster but currently far enough behind
                        # we check if it can maintain a gap of min_time_gap_s by breaking with decel_max
                        t = np.arange(0, follower_rel_speed/follower_decel_max+0.1, 0.1)
                        time_gap = (
                            (follower_distance - follower_rel_speed*t + 0.5 * follower_decel_max*t**2) /
                            (follower_rel_speed + ego_speed - follower_decel_max*t)
                        )
                        safety_margin_behind = np.min(time_gap) > min_time_gap_s
                    else:
                        # follower is slower and currently far enough behind
                        safety_margin_behind = True
                else:
                    # follower is too close
                    safety_margin_behind = False
            else:
                # no following vehicle
                safety_margin_behind = True

            if safety_margin_ahead and safety_margin_behind:
                return "It is safe to merge to this lane.\n"
            else:
                return "It is not safe to merge to this lane.\n"

        vehicles = features.agents[0]
        lines = features.centerlines[0]
        traffic_cones = features.traffic_cones[0]
        ego_speed = np.round(features.ego_speed[0], decimals=decimals)[0]
        speed_limit = np.round(lines.current.speed_limit, decimals=decimals)
        np.set_printoptions(suppress=True)
        user_prompt = (
            f"You are driving at {ego_speed} meters per second on the current lane.\n"
            f"The speed limit is {speed_limit} meters per second.\n"
            "You have the following options:\n"
        )
        user_prompt += _get_centerline_description(lines.current, "current")
        if traffic_cones.front_current is not None:
            long_dist = np.round(np.abs(traffic_cones.front_current[0]), decimals=decimals)
            user_prompt += f" which is blocked by a traffic cone {long_dist} meters ahead.\n"
        elif vehicles.leading_stationary_agent is not None:
            long_dist = np.round(np.abs(vehicles.leading_stationary_agent[0]), decimals=decimals)
            required_offset_abs = np.abs(vehicles.leading_stationary_agent[-1]) + 0.3
            required_offset = np.round(required_offset_abs * np.sign(vehicles.leading_stationary_agent[-1]), decimals=decimals)
            user_prompt += f" which is blocked by a stationary car {long_dist} meters ahead."
            user_prompt += f" Passing it requires and offset of {required_offset} meters from the centerline.\n"
        elif vehicles.front_current is None:
            user_prompt += " without a vehicle ahead.\n"
        else:
            user_prompt += f" and follow {_get_vehicle_description(vehicles.front_current, ego_speed)}.\n"
        
        for lane_role in ["left", "right"]:
            if getattr(lines, lane_role) is None:
                continue
            # lane is not None, so it exists
            user_prompt += _get_centerline_description(getattr(lines, lane_role), lane_role)
            leading_vehicle = getattr(vehicles, "front_"+lane_role)
            following_vehicle = getattr(vehicles, "rear_"+lane_role)
            traffic_cone_ahead = getattr(traffic_cones, "front_"+lane_role)
            if traffic_cone_ahead is not None:
                long_dist = np.round(np.abs(traffic_cone_ahead[0]), decimals=decimals)
                user_prompt += f" which is blocked by a traffic cone {long_dist} meters ahead.\n"
            elif leading_vehicle is None and following_vehicle is None:
                user_prompt += " where currently no vehicles are driving\n"
            elif following_vehicle is None:
                user_prompt += f" and follow {_get_vehicle_description(leading_vehicle, ego_speed)}.\n"
                user_prompt += _get_safety_evaluation(leading_vehicle, None, ego_speed)
            elif leading_vehicle is None:  
                user_prompt += f" and merge before {_get_vehicle_description(following_vehicle, ego_speed)}.\n"
                user_prompt += _get_safety_evaluation(None, following_vehicle, ego_speed)
            else:
                user_prompt += f" and merge between {_get_vehicle_description(leading_vehicle, ego_speed)} and {_get_vehicle_description(following_vehicle, ego_speed)}.\n"
                user_prompt += _get_safety_evaluation(leading_vehicle, following_vehicle, ego_speed)
        user_prompt += _get_route_description(lines)
        user_prompt += (
            "Analyze the traffic situation and give a list of notable vehicles, pedestrians and cyclists in the scenario as well as notable objects.\n"
            "Then analyze each of the lane options regarding safety and progress.\n" 
            "Then decide on the best option. Finally, return the json description mentioned above."
        )

        formatted_system_prompt = f"{self._bos_token}{self._instruction_start_token} {self._system_promt_start_token}\n{self._get_system_prompt()}\n{self._system_promt_end_token}\n"
        formatted_examples_prompt = [
            f"{message} {self._instruction_end_token} {answer} {self._eos_token}{self._bos_token}{self._instruction_start_token} " 
            for message, answer in self._get_examples()
        ]
        formatted_examples_prompt = "".join(formatted_examples_prompt)
        formatted_user_prompt = f"{user_prompt} {self._instruction_end_token} "

        return formatted_system_prompt+formatted_examples_prompt+formatted_user_prompt

class DummyBehaviorPlanner(AbstractLLMBehaviorPlanner):
    def __init__(
        self,
        strategy: str,
    ) -> None:
        SUPPORTED_STRATEGIES = ["random", "current", "left", "right"]
        assert strategy in SUPPORTED_STRATEGIES, f"strategy has to be in {SUPPORTED_STRATEGIES}"
        self._strategy = strategy

    def initialize(self, initialization: PlannerInitialization) -> None:
        pass

    def infer_behavior(self, features: LLMFeatures) -> LaneFollowingBehavior:
        centerlines=features.centerlines[0]
        if self._strategy == "random":
            target_centerline_key = np.random.choice(["current","left","right"])
        else:
            target_centerline_key = self._strategy
        
        if getattr(centerlines,target_centerline_key):
            centerline = getattr(centerlines,target_centerline_key).poses
        else:
            # if desired centerline is not available, use current centerline as fallback
            centerline = centerlines.current.poses
        return LaneFollowingBehavior(
            centerline=centerline,
            centerline_offset=0.0,
            speed_limit=14.0,
        )

    def get_comment_on_current_behavior(self) -> str:
        return f"infered lane according to strategy {self._strategy}"

    def get_prompt_for_current_behavior(self) -> str:
        return f"dummy: infer the lane according to strategy {self._strategy}"
