from functools import cached_property
import logging
from typing import List, Type
import time
import numpy as np
import ast
import warnings

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from tuplan_garage.planning.simulation.planner.llm_planner.llm_interface.llm_interface import AbstractLLMInterface
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
from tuplan_garage.planning.training.preprocessing.feature_builders.gpt_driver_feature_builder import GPTFeatureBuilder
from tuplan_garage.planning.training.preprocessing.features.gtp_driver_features import GPTDriverFeatures
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states


logger = logging.getLogger(__name__)

class GPTDriverWaypointsPlanner(AbstractPlanner):
    def __init__(
            self,
            gpt_feature_builder: GPTFeatureBuilder,
            infer_llm_every_n_iterations: int,
            llm_interface: AbstractLLMInterface,
        ):
        self.feature_builder = gpt_feature_builder
        self.llm_interface = llm_interface
        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []
        self._infer_llm_every_n_iterations = infer_llm_every_n_iterations

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore
    
    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__
    
    def get_comment_on_current_behavior(self) -> str:
        return self._current_response

    def get_prompt_for_current_behavior(self) -> str:
        return self.system_prompt + "\n\n" + self._current_prompt

    @cached_property
    def system_prompt(self) -> str:
        return (
            "Autonomous Driving Planner\n"
            "Role: You are the brain of an autonomous vehicle. Plan a safe 8-second driving trajectory. Avoid collisions with other objects.\n"
            "\n"
            "Context\n"
            "- Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0).\n"
            "- Heading Angle (theta): The heading angle is with respect to the Z-axis. It's 0 when you're facing the positive X-axis.\n"
            "- Objective: Create a 8-second route using 16 waypoints, one every 0.5 seconds.\n"
            "\n"
            "Inputs\n"
            "1. Perception: Info about surrounding objects.\n"
            "2. Historical Trajectory: Your past 2-second route, given by 4 waypoints.\n"
            "3. Ego-States: Your current state including velocity, heading angular velocity, can bus data, heading speed, and steering signal.\n"
            "4. Mission Goal: Goal location for the next 8 seconds.\n"
            "\n"
            "Task\n"
            "- Thought Process: Note down critical objects and potential effects from your perceptions and predictions.\n"
            "- Action Plan: Detail your meta-actions based on your analysis.\n"
            "- Trajectory Planning: Develop a safe and feasible 8-second route using 16 new waypoints.\n"
            "\n"
            "Output\n"
            "- Thoughts:\n"
            " - Notable Objects\n"
            "   Potential Effects\n"
            "- Meta Action\n"
            "- Trajectory (MOST IMPORTANT):\n"
            "  - [(x1,y1,heading1), (x2,y2,heading2), ... , (x16,y16,heading16)]\n"
        )

    def _build_task_prompt(self, features: GPTDriverFeatures) -> str:
        perception_prompt = "Perception\n"
        for vehicle in features.agents[0]:
            x,y = vehicle[0], vehicle[1]
            theta = vehicle[2]
            vx, vy = vehicle[3], vehicle[4]
            perception_prompt += f"- Vehicle: (x,y,theta,vx,vy): ({x:.1f},{y:.1f},{theta:.1f},{vx:.1f},{vy:.1f})\n"
        for object in features.objects[0]:
            x,y = object[0], object[1]
            theta = object[2]
            perception_prompt += f"- Generic Object: (x,y,theta): ({x:.1f},{y:.1f},{theta:.1f})\n"
        if len(features.objects[0]) == 0 and len(features.agents[0]) == 0:
            perception_prompt += (
                "- No objects or agents in the scene\n"
            )
        dynamic_state = features.ego_state[0]
        vx, vy = dynamic_state[0], dynamic_state[1]
        angular_velocity = dynamic_state[2]
        ax, ay = dynamic_state[3], dynamic_state[4]
        ego_states_prompt = (
            "Ego-States\n"
            f"- Velocity (vx,vy): ({vx:.1f},{vy:.1f})\n"
            f"- Heading Angular Velocity: {angular_velocity:.1f}\n"
            f"- Acceleration (ax,ay): ({ax:.1f},{ay:.1f})\n"
        )
        past_traj = ", ".join([f"({past_pose[0]:.1f},{past_pose[1]:.1f})" for past_pose in features.ego_poses[0][:-1]])
        historical_trajectory_prompt = (
            f"Historical Trajectory (last two seconds): [{past_traj}]"
        )
        # add information, which lanes are on route
        on_route_lanes = features.centerlines[0].keys()
        maneuvers = {
            "left": "merge left",
            "right": "merge right",
            "current": "stay in current lane"
        }
        on_route_maneuvers = [maneuvers[lane] for lane in on_route_lanes]
        on_route_maneuvers_prompt = " or ".join(on_route_maneuvers)
        mission_goal_prompt = (
           f"Mission Goal: {on_route_maneuvers_prompt}\n"
        )
        return (
            perception_prompt
            + "\n"
            + ego_states_prompt
            + "\n"
            + historical_trajectory_prompt
            + "\n"
            + mission_goal_prompt
        )
    
    def initialize(self, initialization: PlannerInitialization) -> None:
        self._initialization = initialization
        self.llm_interface.initialize()
        self._current_response = "Not set"
        self._current_prompt = "Not set"
        self._iteration = 0
    
    def _parse_line_to_numpy(self, line:str) -> np.ndarray:
        return np.array(
            ast.literal_eval(line.lstrip().rstrip())
        )

    def _parse_output_to_trajectory(self, output: str, current_input: PlannerInput) -> InterpolatedTrajectory:
        output_lines = [l.lower() for l in output.split("\n")]
        assert "trajectory:" in output_lines or "trajectory planning:" in output_lines,(
            f"'trajectory:' is not in output {output} with lines {output_lines}"
        )
        # try parsing of the 'trajectory' output
        try:
            traj_idx = output_lines.index("trajectory:") + 1
            waypoints_output = output_lines[traj_idx]
            trajectory = self._parse_line_to_numpy(waypoints_output)
        except Exception:
            # in case it fails, try with 'trajectory planning' output
            try:
                traj_idx = output_lines.index("trajectory planning:") + 1
                waypoints_output = output_lines[traj_idx]
                trajectory = self._parse_line_to_numpy(waypoints_output)
            except Exception:
                raise ValueError(f"Could not parse output to trajectory: {output}")
        
        # make sure the trajectory shape is correct
        if trajectory.shape[-1] == 2:
            heading = np.zeros_like(trajectory[...,:1])
            trajectory = np.concatenate([trajectory, heading], axis=-1)
        elif trajectory.shape[-1] > 3:
            trajectory = trajectory[...,:3]
        try:
            time_horizon = trajectory.shape[0] * 0.5
            absolute_states = transform_predictions_to_states(
                trajectory, current_input.history.ego_states, time_horizon, 0.5
            )
        except Exception as e:
            raise ValueError(f"Could not transform trajectory array of shape {trajectory.shape}, {trajectory}, {output}, {e}")
        return InterpolatedTrajectory(absolute_states)
    
    def _generate_constant_velocity_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        timesteps = np.arange(0, 8.0, 0.5) + 0.5
        speed = current_input.history.ego_states[-1].dynamic_car_state.speed
        x = timesteps * speed
        y = np.zeros_like(x)
        heading = np.zeros_like(x)
        trajectory = np.stack([x,y,heading], axis=-1)

        absolute_states = transform_predictions_to_states(
            trajectory, current_input.history.ego_states, 8.0, 0.5
        )
        return InterpolatedTrajectory(absolute_states)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        if self._iteration % self._infer_llm_every_n_iterations == 0:
            # generate LLM features
            start_time = time.perf_counter()
            gpt_features: GPTDriverFeatures = self.feature_builder.get_features_from_simulation(
                current_input=current_input,
                initialization=self._initialization
                )
            self._feature_building_runtimes.append(time.perf_counter() - start_time)

            # infer LLM
            start_time = time.perf_counter()
            task_prompt = self._build_task_prompt(gpt_features)
            self._current_prompt = task_prompt
            response = self.llm_interface.infer_model(
                system_prompt=self.system_prompt,
                in_context_examples=None,
                task_prompt=task_prompt,
                max_new_tokens=512,
            )
            # trajectory is in absolute coordinates
            try:
                trajectory = self._parse_output_to_trajectory(response, current_input)
                self._current_response = response + "\n\n" + "Trajectory could be parsed"
            except Exception as e:
                warnings.warn(f"Encountered the following Exception during parsing of the output. Using CV as fallback. {e}")
                trajectory = self._generate_constant_velocity_trajectory(current_input)
                self._current_response = response + "\n\n" + f"Constant Velocity Fallback\n {e}"
            self.absolute_trajectory = trajectory
            self._inference_runtimes.append(time.perf_counter() - start_time)
        else:
            trajectory = self.absolute_trajectory

        self._iteration += 1
        return trajectory
