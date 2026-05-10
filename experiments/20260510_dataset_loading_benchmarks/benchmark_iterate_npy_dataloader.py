from multiprocessing import freeze_support
import time

from torch.utils.data import DataLoader

from src_ours.multi_folder_dataset import MultiFolderDataset, MultiFolderNpyDataset


def benchmark_dataloader(loader, name, limit_batches=16):
    print(f"\nBenchmarking: {name}")

    start = time.perf_counter()

    batch_count = 0
    sample_count = 0

    for batch in loader:
        batch_count += 1

        first_key = next(iter(batch.keys()))
        batch_size = batch[first_key].shape[0]

        sample_count += batch_size

        if batch_count >= limit_batches:
            break

    elapsed = time.perf_counter() - start

    print(f"Batches: {batch_count}")
    print(f"Samples: {sample_count}")
    print(f"Time: {elapsed:.3f} sec")
    print(f"Batches/sec: {batch_count / elapsed:.2f}")
    print(f"Samples/sec: {sample_count / elapsed:.2f}")

def main():
    train_dataset = MultiFolderNpyDataset(
        root="data/embed2heights_npy",
        split="train",
        input_folders=[
            "alphaearth_emb",
            "terramind_s1_emb",
            "terramind_s2_emb",
            "tessera_emb",
            "thor_s1_emb",
            "thor_s2_emb",
        ],
    )

    test_dataset = MultiFolderNpyDataset(
        root="data/embed2heights_npy",
        split="test",
        input_folders=[
            "alphaearth_test_emb",
            "terramind_test_s1_emb",
            "terramind_test_s2_emb",
            "tessera_test_emb",
            "thor_test_s1_emb",
            "thor_test_s2_emb",
        ],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    benchmark_dataloader(train_loader, "train npy", limit_batches=32)
    benchmark_dataloader(test_loader, "test npy", limit_batches=32)

    # Benchmarking: train npy
    # Batches: 32
    # Samples: 128
    # Time: 21.750 sec
    # Batches/sec: 1.47
    # Samples/sec: 5.88

    # Benchmarking: test npy
    # Batches: 32
    # Samples: 128
    # Time: 15.648 sec
    # Batches/sec: 2.04
    # Samples/sec: 8.18

    del train_dataset; del train_loader; del test_dataset; del test_loader

    train_dataset_non_npy = MultiFolderDataset(
        root="data/embed2heights/data",
        split="train",
        input_folders=[
            "alphaearth_emb",
            "terramind_s1_emb",
            "terramind_s2_emb",
            "tessera_emb",
            "thor_s1_emb",
            "thor_s2_emb",
        ],
    )

    test_dataset_non_npy = MultiFolderDataset(
        root="data/embed2heights/data",
        split="test",
        input_folders=[
            "alphaearth_test_emb",
            "terramind_test_s1_emb",
            "terramind_test_s2_emb",
            "tessera_test_emb",
            "thor_test_s1_emb",
            "thor_test_s2_emb",
        ],
    )


    train_loader_non_npy = DataLoader(
        train_dataset_non_npy,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader_non_npy = DataLoader(
        test_dataset_non_npy,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    benchmark_dataloader(train_loader_non_npy, "train tif", limit_batches=12)
    benchmark_dataloader(test_loader_non_npy, "test tif", limit_batches=12)

    # Benchmarking: train tif
    # Batches: 12
    # Samples: 48
    # Time: 24.567 sec
    # Batches/sec: 0.49
    # Samples/sec: 1.95

    # Benchmarking: test tif
    # Batches: 12
    # Samples: 48
    # Time: 30.393 sec
    # Batches/sec: 0.39
    # Samples/sec: 1.58


if __name__ == "__main__":
    freeze_support()
    main()