import pathlib
from dataclasses import dataclass
from typing import List, Union

from nuplan.common.utils.s3_utils import is_s3_path
from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

from tuplan_garage.planning.simulation.planner.llm_planner.llm_planner import LLMPlanner

@dataclass
class LLMResponse:
    iteration_index: int
    prompt: str
    response: str

    def __eq__(self, other):
        if not isinstance(other, LLMResponse):
            return False
        return self.prompt == other.prompt and self.response == other.response

class LLMResponseLoggingCallback(AbstractCallback):
    """
    Base class for simulation callbacks.
    """
    def __init__(
        self,
        output_directory: Union[str, pathlib.Path],
        llm_responses_log_dir: Union[str, pathlib.Path],
    ):
        self._output_directory = pathlib.Path(output_directory) / llm_responses_log_dir

        self._llm_responses: List[LLMResponse] = []

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        Called when initialization of simulation starts.
        :param setup: simulation setup
        :param planner: planner before initialization
        """
        scenario_directory = self._get_scenario_folder(planner.name(), setup.scenario)

        if not is_s3_path(scenario_directory):
            scenario_directory.mkdir(exist_ok=True, parents=True)

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        Called when initialization of simulation ends.
        :param setup: simulation setup
        :param planner: planner after initialization
        """
        pass

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        Called when simulation step starts.
        :param setup: simulation setup
        :param planner: planner at start of a step
        """
        pass

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """
        Called when simulation step ends.
        :param setup: simulation setup
        :param planner: planner at end of a step
        :param sample: result of a step
        """
        pass

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """
        Called when planner starts to compute trajectory.
        :param setup: simulation setup
        :param planner: planner before planner.compute_trajectory() is called
        """
        pass

    def on_planner_end(self, setup: SimulationSetup, planner: LLMPlanner, trajectory: AbstractTrajectory) -> None:
        """
        Called when planner ends to compute trajectory.
        :param setup: simulation setup
        :param planner: planner after planner.compute_trajectory() is called
        :param trajectory: trajectory resulting from planner
        """
        item = LLMResponse(
            iteration_index=setup.time_controller.get_iteration().index,
            prompt=planner.get_prompt_for_current_behavior(),
            response=planner.get_comment_on_current_behavior(),
        )
        if len(self._llm_responses) == 0 or self._llm_responses[-1] != item:
            self._llm_responses.append(item)

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """
        Called when simulation starts.
        :param setup: simulation setup
        """
        pass

    def on_simulation_end(self, setup: SimulationSetup, planner: AbstractPlanner, history: SimulationHistory) -> None:
        """
        Called when simulation ends.
        :param setup: simulation setup
        :param planner: planner when simulation ends
        :param history: resulting from simulation
        """
        # Generate table
        html_table = _generate_html_table(self._llm_responses, setup.scenario.scenario_name)
        
        # Create directory
        scenario_directory = self._get_scenario_folder(planner.name(), setup.scenario)
        file_name = scenario_directory / (setup.scenario.scenario_name + ".html")

        # write html to file
        with open(file_name, 'w') as file:
            file.write(html_table)

    def _get_scenario_folder(self, planner_name: str, scenario: AbstractScenario) -> pathlib.Path:
        """
        Compute scenario folder directory where all files will be stored.
        :param planner_name: planner name.
        :param scenario: for which to compute directory name.
        :return directory path.
        """
        return self._output_directory / planner_name / scenario.scenario_type / scenario.log_name / scenario.scenario_name  # type: ignore


def _generate_html_table(items: List[LLMResponse], scenario_name: str) -> str:
    def _escape_html_chars(line: str) -> str:
        return line.replace("<","&lt;").replace(">","&gt;").replace("\"","&quot;")

    table_style = """
        <style>
            h1 {
                font-weight: lighter;
                text-align: center;
            }
            table {
                border-collapse: collapse;
                width: 80%;
                margin: 20px auto;
                font-family: monospace;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
                white-space: pre-wrap;
                text-align: left;
                vertical-align: top;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            tr:hover {
                background-color: #e6e6e6;
            }
            th, td {
                transition: 0.3s;
            }
            th:hover {
                background-color: #dcdcdc;
            }
        </style>
    """
    table_rows = "".join(
        [
            f"<tr><td><p>{entry.iteration_index}</p></td><td><p>{_escape_html_chars(entry.prompt)}</p></td><td><p>{_escape_html_chars(entry.response)}</p></td></tr>"
            for entry in items
        ]
    )

    table = table_style + f"""
        <h1>
            Overview for scenario {scenario_name}
        </h1>
        <table>
            <tr>
                <th>Index</th>
                <th>Prompt</th>
                <th>Response</th>
            </tr>
            {table_rows}
        </table>
        """
    return table
