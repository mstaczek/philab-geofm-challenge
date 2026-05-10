# Experiments history

## 20260510 Comparing tif vs npy dataset format

npy data is preprocessed tif:
- removed nans,
- added padding so that all samples not shaped 16x16 or 256x256 will be 16x16 or 256x256
- data saved as float16 after clipping.

Data size was reduced by half by changing float32 to float16. Still, the improvement is more than x2.

### torch DataLoaders

Parameters:
- batch size 4,
- 2 workers.

| Dataset | Format | Batches | Samples | Time (sec) | Batches/sec | Samples/sec |
|---|---|---:|---:|---:|---:|---:|
| Train | NPY | 32 | 128 | 21.750 | 1.47 | 5.88 |
| Test | NPY | 32 | 128 | 15.648 | 2.04 | 8.18 |
| Train | TIF | 12 | 48 | 24.567 | 0.49 | 1.95 |
| Test | TIF | 12 | 48 | 30.393 | 0.39 | 1.58 |

Improvement x4 - tif format is 4 times slower than npy.

### torch Datasets

| Dataset | Format | Samples | Time (sec) | Samples/sec |
|---|---|---:|---:|---:|
| Train | NPY | 2024 | 230.147 | 8.79 |
| Test | NPY | 946 | 128.030 | 7.39 |
| Train | TIF | 128 | 96.060 | 1.33 |
| Test | TIF | 128 | 126.120 | 1.01 |

Improvement x7 - tif format is 4 times slower than npy.

