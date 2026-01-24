from dataclasses import dataclass

from spino.constants import MODELS_ROOT, FIGURES_ROOT, RUNS_ROOT


@dataclass
class PathConfig:

    experiment_type_name: str

    def __post_init__(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> str:
        return RUNS_ROOT / self.experiment_type_name

    @property
    def model_dir(self) -> str:
        return MODELS_ROOT / self.experiment_type_name

    @property
    def figure_dir(self) -> str:
        return FIGURES_ROOT / self.experiment_type_name
