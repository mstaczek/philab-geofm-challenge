from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] # add repo root (2 folder above this script) to python path
sys.path.insert(0, str(ROOT))

from src_ours.models import DummyModelDictInput

from core.utils import get_torch_device
from src_ours.run_training import run_training 
from src_ours.constants import SOURCE_ROOT_NPY


def main():
    model = DummyModelDictInput()

    run_training(
        data_root=SOURCE_ROOT_NPY,
        experiment_name="test_dummy-saving-fixing",
        output_dir="runs",
        batch_size=128,
        epochs=1,
        device=get_torch_device("cuda"),
        random_seed=42,
        dataset_names=["terraminds1"],
        model=model,
        save_zip=True,
        zip_output_name="testing_to_delete-4.zip",
    )

if __name__ == "__main__":
    main()

