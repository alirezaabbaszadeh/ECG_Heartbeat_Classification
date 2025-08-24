# ECG Heartbeat Classification

## Clinical Relevance
Electrocardiogram (ECG) analysis is central to detecting arrhythmias and other cardiac abnormalities. Accurate heartbeat classification helps clinicians identify life‑threatening conditions early, supports continuous patient monitoring, and guides timely intervention.

## Repository Overview
This repository investigates ECG heartbeat classification with a modern Conformer‑based architecture. It includes scripts for dataset preparation, focused hyperparameter tuning, and a comprehensive evaluation pipeline:

- **Hyperparameter tuning** – `run_hyperparameter_tuning.py` explores model configurations to optimize performance.
- **Conformer architecture** – `ModelBuilder.py` defines the CNN‑Conformer model and related ablation variants.
- **Evaluation workflow** – `run_kfold_evaluation.py` and `run_final_evaluation.py` provide reproducible k‑fold and held‑out evaluations.

## Installation

1. Ensure **Python 3.10 or newer** is installed. TensorFlow 2.20 requires Python ≥3.10.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Upgrade pip and install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. For GPU acceleration, install the NVIDIA driver and matching CUDA and cuDNN libraries for TensorFlow 2.20 (e.g., CUDA 12.2 and cuDNN 9.0). Refer to the official TensorFlow GPU documentation for details.

## Data Acquisition

The experiments draw on the **MIT-BIH Arrhythmia Database**, a benchmark collection of 48 half‑hour two‑lead ambulatory ECG recordings originally published by the BIH Arrhythmia Laboratory and now distributed via PhysioNet [1]. The archive provides beat‑level annotations curated by expert electrophysiologists and is widely used for evaluating arrhythmia detection algorithms.

Download the raw signals and annotations with the provided helper script:

```bash
python download_data.py
```

By default the script retrieves the entire database (≈110 MB) and stores it in `mit-bih-arrhythmia-database-1.0.0/` adjacent to the repository root. After completion the directory contains one triplet of files per record:

```
mit-bih-arrhythmia-database-1.0.0/
├── 100.atr
├── 100.dat
├── 100.hea
├── …
├── 234.atr
├── 234.dat
├── 234.hea
└── README
```

Modify the `db_name` and `save_directory` variables in `download_data.py` to target alternative PhysioNet databases or custom storage locations.

## Preprocessing

Convert the raw MIT‑BIH records into model‑ready examples with a two‑stage preprocessing pipeline. Run the scripts in sequence after downloading the database.

### 1. Generate scalograms and beat metadata

```bash
python preprocess_data.py
```

This script isolates 187‑sample beats, computes Morlet wavelet scalograms, and writes one HDF5 file per record to `preprocessed_data_h5_raw/`. A companion `metadata.json` in the same directory records, for every beat, its source record, index, and AAMI class label. Signals are deliberately kept **unnormalized** at this stage to avoid train–test leakage; normalization is applied later during dataset loading using statistics derived from the training set.

### 2. Package sequences into batched TFRecords

```bash
python create_batched_tfrecords.py
```

Using the stored metadata, this script forms overlapping sequences of three consecutive beats and serializes them in batches of 256 to `tfrecord_data_batched/`. Each TFRecord example embeds the batch size, sequence length, and scalogram dimensions alongside the raw byte arrays, producing a self‑describing format that streams efficiently through `tf.data`.

### Why batched TFRecords?

Batched TFRecords amortize file‑open overhead and permit large sequential reads, significantly accelerating I/O compared with per‑example files. When coupled with `tf.data` interleave and prefetch operations, the batched layout sustains high throughput on both CPUs and GPUs.

### Normalization strategy

Deferring normalization until dataset construction ensures that scaling parameters are computed exclusively from the training split. This on‑the‑fly approach preserves the integrity of validation and test sets and aligns with best practices for reproducible research.

## Dataset Loading

`DataLoader.py` assembles streaming datasets directly from the batched TFRecords and employs a two‑stage shuffling scheme that maximizes example diversity without incurring large memory overhead:

1. **File‑level shuffle** – when `is_training=True`, the list of TFRecord files is shuffled globally so that each epoch traverses records in a different order.
2. **Interleave shuffle** – `tf.data.Dataset.interleave` reads from several files concurrently (default `cycle_length=4`), mixing individual examples across files to disrupt local ordering.

After decoding, an optional light shuffle with a small buffer adds an additional layer of randomness before batching.

The loader also supports **on‑the‑fly normalization**. If per‑channel `mean` and `scale` arrays are supplied, they are converted to tensors and each scalogram is standardized as `(scalogram - mean) / scale` before being expanded to include a channel dimension.

```python
from DataLoader import create_dataset, get_all_labels

config = {...}  # paths and preprocessing parameters
record_names = ["100", "101", "102"]
mean, scale = train_mean, train_std  # tensors or NumPy arrays

train_ds = create_dataset(record_names, config, batch_size=32,
                          is_training=True, mean=mean, scale=scale)
for scalograms, labels in train_ds.take(1):
    pass  # training loop or debugging
```

## Model Architectures

### Conformer-Based Network

The primary model, implemented in `ModelBuilder.py`, couples a convolutional front end with stacked **Conformer** blocks (Figure 1 in [2]). Each scalogram passes through a 2‑D CNN feature extractor before being processed by a sequence of Conformer blocks that integrate three complementary modules:

- **Feed‑Forward Module** – two half‑step feed‑forward networks with Swish activation and dropout surround the attention and convolution layers, following the Macaron design.
- **Multi‑Head Self‑Attention** – captures global temporal context; positional embeddings are added before attention to encode beat order.
- **Convolution Module** – a depthwise separable 1‑D convolution with gated linear units and batch normalization models local dependencies and sharp morphological patterns.

```text
Input → ½ Feed‑Forward → Self‑Attention → Convolution → ½ Feed‑Forward → LayerNorm → Output
```

### Ablation Models

- **Attention‑Only** – removes the convolution module, yielding a Transformer‑style encoder used to quantify the incremental value of local convolutions.
- **CNN‑LSTM** – replaces Conformer blocks with a unidirectional LSTM after the CNN extractor, providing a classical recurrent baseline for temporal modeling.

```text
Attention‑Only:  Input → [Self‑Attention → Feed‑Forward] × N → Output
CNN‑LSTM:       Input scalograms → Time‑Distributed CNN → LSTM → Output
```

These controlled variants isolate the contribution of attention and convolution mechanisms, enabling rigorous scientific evaluation of architectural choices.

## Training

### Hyperparameter tuning

Explore model configurations with `run_hyperparameter_tuning.py`. The script accepts a `--model_name` argument to choose among the implemented architectures【F:run_hyperparameter_tuning.py†L134-L141】:

```bash
python run_hyperparameter_tuning.py --model_name Main_Model
```

Internally, the script partitions the available record names with scikit‑learn’s `KFold`, yielding fold‑specific training and validation lists that serve as input to the `TimeSeriesModel` pipeline【F:run_hyperparameter_tuning.py†L177-L183】.

### K‑fold strategy

`TimeSeriesModel` operates on a single fold at a time. For each fold it computes normalization statistics exclusively on the training subset, builds `tf.data` streams, and launches a Hyperband tuner, ensuring that hyperparameter search is free from data leakage【F:MainClass.py†L34-L91】【F:MainClass.py†L125-L149】.

### Tuning artifacts

Every tuning run creates a timestamped directory under `Research_Runs/` containing logs, data splits, and configuration files【F:run_hyperparameter_tuning.py†L72-L77】【F:run_hyperparameter_tuning.py†L154-L174】. Fold‑specific assets are stored deeper in `Research_Runs/run_<timestamp>/fold_<n>/`, and the script copies the best hyperparameter set to the top‑level run folder for convenient reuse【F:MainClass.py†L93-L97】【F:run_hyperparameter_tuning.py†L224-L229】.

## Evaluation

### K-fold cross-validation

Assess generalization by re‑using the tuned hyperparameters across stratified folds:

```bash
python run_kfold_evaluation.py --model_name Main_Model
```

The script writes fold‑specific histories and a summary of mean and standard‑deviation metrics to `Research_Runs/kfold_eval_<model>_<timestamp>/`【F:run_kfold_evaluation.py†L43-L47】【F:run_kfold_evaluation.py†L180-L183】. Use the aggregated statistics to compare models; significance can be examined via paired tests (e.g., t‑test or Wilcoxon signed‑rank) applied to matching fold metrics.

### Final held‑out evaluation

Train the champion model on all training records and evaluate on the untouched test set:

```bash
python run_final_evaluation.py --model_name Main_Model
```

Optional `--epochs` and `--batch_size` arguments override training length and batch size【F:run_final_evaluation.py†L137-L141】. Artifacts—including the final model and evaluation reports—are saved under `Research_Runs/final_run_<model>_<timestamp>/`【F:run_final_evaluation.py†L38-L43】【F:run_final_evaluation.py†L222-L236】.

### Evaluator outputs

`Evaluator.py` produces comprehensive diagnostics for each evaluation run:

- `classification_report.txt` with per‑class precision, recall, F1‑score, and AUC values【F:Evaluator.py†L140-L148】
- `confusion_matrix.png` visualizing correct versus incorrect classifications【F:Evaluator.py†L150-L160】
- `roc_curves.png` plotting one‑vs‑rest ROC curves with corresponding AUCs【F:Evaluator.py†L162-L178】

These files are written to the same run directory passed to `Evaluator.save_results`.

### Interpreting metrics and significance

Precision, recall, and F1‑score contextualize performance under class imbalance, while ROC/AUC summarizes discrimination across thresholds; the confusion matrix highlights systematic misclassifications. Report mean ± standard deviation across folds or repeated runs to convey variability. When comparing models, apply statistical tests on per‑fold metrics to determine whether observed differences exceed random variation (commonly using a significance threshold of *p* < 0.05).

## Repository Structure
- **Data preparation** – `download_data.py`, `preprocess_data.py`, and `create_batched_tfrecords.py` fetch the MIT‑BIH arrhythmia dataset, perform signal cleaning, and package examples into TFRecord batches.
- **Core modules** – `ModelBuilder.py`, `DataLoader.py`, and `Evaluator.py` implement the model architecture, streaming data pipeline, and evaluation metrics.
- **Run scripts** – automation helpers such as `run_full_pipeline.sh`, `run_hyperparameter_tuning.py`, `run_kfold_evaluation.py`, and `run_final_evaluation.py` orchestrate training, tuning, and assessment.
- **Research artifacts** – experiment outputs and logs are stored in `Research_Runs/`.

## Reproducible Research
Consistent configuration files, deterministic seeds, and scripted training/evaluation workflows align the project with reproducible research practices, enabling others to replicate and extend the results.

## References

[1] G. B. Moody and R. G. Mark, “The impact of the MIT-BIH Arrhythmia Database,” *IEEE Engineering in Medicine and Biology Magazine*, vol. 20, no. 3, pp. 45–50, 2001. doi:10.1109/51.932724
[2] A. Gulati, J. Qin, C.-C. Chiu, et al., “Conformer: Convolution-augmented Transformer for Speech Recognition,” in *Proceedings of Interspeech*, 2020, pp. 5036–5040. doi:10.21437/Interspeech.2020-3015
