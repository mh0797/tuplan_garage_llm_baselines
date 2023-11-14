from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor
)

@dataclass
class SurroundingVehicles():
    front_right: FeatureDataType
    rear_right: FeatureDataType
    front_left: FeatureDataType
    rear_left: FeatureDataType
    front_current: FeatureDataType
    rear_current: FeatureDataType
    leading_stationary_agent: FeatureDataType
    oncoming_agent: FeatureDataType

    def to_feature_tensor(self) -> SurroundingVehicles:
        return SurroundingVehicles(
            front_right=to_tensor(self.front_right),
            rear_right=to_tensor(self.rear_right),
            front_left=to_tensor(self.front_left),
            rear_left=to_tensor(self.rear_left),
            front_current=to_tensor(self.front_current),
            rear_current=to_tensor(self.rear_current),
            leading_stationary_agent=to_tensor(self.leading_stationary_agent),
            oncoming_agent=to_tensor(self.oncoming_agent),
    )

    def to_device(self, device: torch.device) -> SurroundingVehicles:
        return SurroundingVehicles(
            front_right=self.front_right.to(device),
            rear_right=self.rear_right.to(device),
            front_left=self.front_left.to(device),
            rear_left=self.rear_left.to(device),
            front_current=self.front_current.to(device),
            rear_current=self.rear_current.to(device),
            leading_stationary_agent=self.leading_stationary_agent.to(device),
            oncoming_agent=self.oncoming_agent.to(device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> SurroundingVehicles:
        """Implemented. See interface."""
        return SurroundingVehicles(
            front_right=data["front_right"],
            rear_right=data["rear_right"],
            front_left=data["front_left"],
            rear_left=data["rear_left"],
            front_current=data["front_current"],
            rear_current=data["rear_current"],
            leading_stationary_agent=data["leading_stationary_agent"],
            oncoming_agent=data["oncoming_agent"],
        )

@dataclass
class AvailableCenterlines():
    left: Centerline
    right: Centerline
    current: Centerline

    def to_feature_tensor(self) -> AvailableCenterlines:
        return AvailableCenterlines(
            left=to_tensor(self.left),
            right=to_tensor(self.right),
            current=to_tensor(self.current),
    )

    def to_device(self, device: torch.device) -> AvailableCenterlines:
        return AvailableCenterlines(
            left=self.left.to_device(device),
            right=self.right.to_device(device),
            current=self.current.to_device(device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AvailableCenterlines:
        """Implemented. See interface."""
        return AvailableCenterlines(
            left=Centerline.deserialize(data["left"]),
            right=Centerline.deserialize(data["right"]),
            current=Centerline.deserialize(data["current"]),
        )

@dataclass
class Centerline():
    id: int
    successor_id: int
    poses: List[FeatureDataType]
    required_lane_changes_to_route: int
    speed_limit: float

    def to_feature_tensor(self) -> Centerline:
        return Centerline(
            id=to_tensor(self.id),
            successor_id=to_tensor(self.successor_id),
            poses=[to_tensor(pose) for pose in self.poses],
            required_lane_changes_to_route=to_tensor(self.required_lane_changes_to_route),
            speed_limit=to_tensor(self.speed_limit),
    )

    def to_device(self, device: torch.device) -> Centerline:
        self.id = torch.tensor(self.id)
        return Centerline(
            id=self.id.to(device),
            successor_id=self.successor_id.to(device),
            poses=[pose.to(device) for pose in self.poses],
            required_lane_changes_to_route=self.required_lane_changes_to_route.to(device),
            speed_limit=self.speed_limit.to(device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Centerline:
        """Implemented. See interface."""
        return Centerline(
            id=data["id"],
            successor_id=data["successor_id"],
            front_left=data["poses"],
            required_lane_changes_to_route=data["required_lane_changes_to_route"],
            speed_limit=data["speed_limit"],
        )

@dataclass
class LLMFeatures(AbstractModelFeature):
    """
    Dataclass that holds map/environment signals that can be consumed by a LLM behavior planner
    All params are wrapped in a list representing the batch dimension
    :param ego_speed: Tensor containing only the current velocity
    :param agents: SurroundingVehicles
    :param centerlines: AvailableCenterlines
    """

    ego_speed: List[FeatureDataType]
    centerlines: List[AvailableCenterlines]
    agents: List[SurroundingVehicles]
    traffic_cones: List[SurroundingVehicles]

    def to_feature_tensor(self) -> AbstractModelFeature:
        """Implemented. See interface."""
        return LLMFeatures(
            ego_speed=[to_tensor(sample) for sample in self.ego_speed],
            centerlines=[to_tensor(sample) for sample in self.centerlines],
            agents=[to_tensor(sample) for sample in self.agents],
            traffic_cones=[to_tensor(sample) for sample in self.traffic_cones],
        ) 

    def to_device(self, device: torch.device) -> LLMFeatures:
        """Implemented. See interface."""
        return LLMFeatures(
            ego_speed=[sample.to(device) for sample in self.ego_speed],
            centerlines=[sample.to_device(device) for sample in self.centerlines],
            agents=[sample.to_device(device) for sample in self.agents],
            traffic_cones=[sample.to_device(device) for sample in self.traffic_cones],
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> LLMFeatures:
        """Implemented. See interface."""
        return LLMFeatures(
            ego_speed=data["ego_speed"],
            centerlines=AvailableCenterlines.deserialize(data["centerlines"]),
            agents=SurroundingVehicles.deserialize(data["agents"]),
            traffic_cones=SurroundingVehicles.deserialize(data["traffic_cones"]),
        )

    def unpack(self) -> List[LLMFeatures]:
        """Implemented. See interface."""
        return [
            LLMFeatures(
                ego_speed=[ego_speed],
                centerlines=[centerlines],
                agents=[agents],
                traffic_cones=[traffic_cones]
            )
            for ego_speed, centerlines, agents, traffic_cones
            in zip(
                self.ego_speed, 
                self.centerlines,
                self.agents,
                self.traffic_cones,
            )
        ]
    
    @classmethod
    def collate(cls, batch: List[LLMFeatures]) -> LLMFeatures:
        return LLMFeatures(
            ego_speed=[sample.ego_speed for sample in batch],
            agents=[sample.agents for sample in batch],
            centerlines=[sample.centerlines for sample in batch],
            traffic_cones=[sample.traffic_cones for sample in batch],
        )