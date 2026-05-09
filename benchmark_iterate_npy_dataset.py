import time

from multi_folder_dataset import MultiFolderNpyDataset


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

sample = train_dataset[0]

for k, v in sample.items():
    print(k, v.shape, v.dtype)
    
def benchmark_dataset(dataset, name, limit=2048):
    print(f"\nBenchmarking: {name}")
    start = time.perf_counter()
    count = 0
    for sample in dataset:
        count += 1
        if count >= limit:
            break

    counted = min(len(dataset), count)

    elapsed = time.perf_counter() - start
    print(f"Samples: {counted}")
    print(f"Time: {elapsed:.3f} sec")
    print(f"Samples/sec: {counted / elapsed:.2f}")

benchmark_dataset(train_dataset, "train npy")
benchmark_dataset(test_dataset, "test npy")

# Benchmarking: train npy
# Samples: 2024
# Time: 230.147 sec
# Samples/sec: 8.79

# Benchmarking: test npy
# Samples: 946
# Time: 128.030 sec
# Samples/sec: 7.39