from typing import List

from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


def load_route_dicts(route_roadblock_ids: List[str], map_api: AbstractMap) -> None:
    """
    Loads roadblock and lane dictionaries of the target route from the map-api.
    :param route_roadblock_ids: ID's of on-route roadblocks
    """

    # remove repeated ids while remaining order in list
    route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

    route_roadblock_dict = {}
    route_lane_dict = {}

    for id_ in route_roadblock_ids:
        block = map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
        block = block or map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)

        route_roadblock_dict[block.id] = block

        for lane in block.interior_edges:
            route_lane_dict[lane.id] = lane

    return route_roadblock_dict, route_lane_dict