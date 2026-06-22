from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] # add repo root (2 folder above this script) to python path
sys.path.insert(0, str(ROOT))

from src_ours.models import DualInputLightUNet

from core.utils import get_torch_device
from src_ours.run_training import run_training 
from src_ours.constants import SOURCE_ROOT_NPY


def main():
    model = DualInputLightUNet(
        main_input_name="alphaearth",
        aux_input_name="terraminds1",
        main_in_channels=64,
        aux_in_channels=768,
    )

    run_training(
        data_root=SOURCE_ROOT_NPY,
        experiment_name="testing_joined_dataset_training_v3",
        output_dir="runs",
        batch_size=4,
        epochs=1,
        device=get_torch_device("cuda"),
        random_seed=42,
        dataset_names=["alphaearth", "terraminds1"],
        model=model,
        save_zip=True,
        zip_output_name="testing_joined_dataset_training_v3.zip",
    )

if __name__ == "__main__":
    main()

