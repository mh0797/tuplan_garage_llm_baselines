import unittest
from hypothesis import strategies as st
from hypothesis import given, settings

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.nuplan_db.test.nuplan_scenario_test_utils import get_test_nuplan_scenario

from tuplan_garage.planning.training.preprocessing.feature_builders.gpt_driver_feature_builder import GPTFeatureBuilder

class TestAgentsFeatureBuilder(unittest.TestCase):
    """Test builder that constructs agent features during training and simulation."""

    def setUp(self) -> None:
        """
        Set up test case.
        """
        self.batch_size = 1
        self.past_time_horizon = 4.0
        self.num_past_poses = 4
        self.num_total_past_poses = self.num_past_poses + 1  # past + present
        self.scenario = get_test_nuplan_scenario()

    @given(
        number_of_detections=st.sampled_from([0, 10]),
        feature_builder_type=st.sampled_from([TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN]),
    )
    @settings(deadline=None)
    def test_agent_feature_builder(self, number_of_detections: int, feature_builder_type: TrackedObjectType) -> None:
        """
        Test AgentFeatureBuilder with and without agents in the scene for both pedestrian and vehicles
        """
        _num_surrounding_agents = 10
        _num_centerlines = 3

        feature_builder = GPTFeatureBuilder(
            num_centerlines=_num_centerlines,
            num_surrounding_agents=_num_surrounding_agents,
            num_surrounding_objects=_num_surrounding_agents,
            centerline_resolution=1.2,
            min_centerline_length=80.0
        )
        feature = feature_builder.get_features_from_scenario(self.scenario)

        assert feature
        assert len(feature.agents) == 1
        assert len(feature.centerlines) == 1
        assert len(feature.agents) == 1

        assert len(feature.agents[0]) <= _num_surrounding_agents
        assert len(feature.centerlines[0]) <= _num_centerlines