import inspect

from src.config import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR
from src.eda.run_eda import run_eda
from src.models.train import run_training


def test_figures_live_in_screenshots():
    assert FIGURES_DIR == PROJECT_ROOT / "screenshots"
    assert REPORTS_DIR == PROJECT_ROOT / "reports"


def test_eda_default_output_paths():
    parameters = inspect.signature(run_eda).parameters
    assert parameters["output_dir"].default == FIGURES_DIR / "EDA"
    assert parameters["reports_dir"].default == REPORTS_DIR


def test_training_default_output_paths():
    parameters = inspect.signature(run_training).parameters
    assert parameters["figures_dir"].default == FIGURES_DIR / "models"
    assert parameters["reports_dir"].default == REPORTS_DIR
