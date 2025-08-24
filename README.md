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

## Repository Structure
- **Data preparation** – `download_data.py`, `preprocess_data.py`, and `create_batched_tfrecords.py` fetch the MIT‑BIH arrhythmia dataset, perform signal cleaning, and package examples into TFRecord batches.
- **Core modules** – `ModelBuilder.py`, `DataLoader.py`, and `Evaluator.py` implement the model architecture, streaming data pipeline, and evaluation metrics.
- **Run scripts** – automation helpers such as `run_full_pipeline.sh`, `run_hyperparameter_tuning.py`, `run_kfold_evaluation.py`, and `run_final_evaluation.py` orchestrate training, tuning, and assessment.
- **Research artifacts** – experiment outputs and logs are stored in `Research_Runs/`.

## Reproducible Research
Consistent configuration files, deterministic seeds, and scripted training/evaluation workflows align the project with reproducible research practices, enabling others to replicate and extend the results.

## References

[1] G. B. Moody and R. G. Mark, “The impact of the MIT-BIH Arrhythmia Database,” *IEEE Engineering in Medicine and Biology Magazine*, vol. 20, no. 3, pp. 45–50, 2001. doi:10.1109/51.932724
