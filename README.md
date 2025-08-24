# ECG Heartbeat Classification

## Clinical Relevance
Electrocardiogram (ECG) analysis is central to detecting arrhythmias and other cardiac abnormalities. Accurate heartbeat classification helps clinicians identify life‑threatening conditions early, supports continuous patient monitoring, and guides timely intervention.

## Repository Overview
This repository investigates ECG heartbeat classification with a modern Conformer‑based architecture. It includes scripts for dataset preparation, focused hyperparameter tuning, and a comprehensive evaluation pipeline:

- **Hyperparameter tuning** – `run_hyperparameter_tuning.py` explores model configurations to optimize performance.
- **Conformer architecture** – `ModelBuilder.py` defines the CNN‑Conformer model and related ablation variants.
- **Evaluation workflow** – `run_kfold_evaluation.py` and `run_final_evaluation.py` provide reproducible k‑fold and held‑out evaluations.

## Repository Structure
- **Data preparation** – `download_data.py`, `preprocess_data.py`, and `create_batched_tfrecords.py` fetch the MIT‑BIH arrhythmia dataset, perform signal cleaning, and package examples into TFRecord batches.
- **Core modules** – `ModelBuilder.py`, `DataLoader.py`, and `Evaluator.py` implement the model architecture, streaming data pipeline, and evaluation metrics.
- **Run scripts** – automation helpers such as `run_full_pipeline.sh`, `run_hyperparameter_tuning.py`, `run_kfold_evaluation.py`, and `run_final_evaluation.py` orchestrate training, tuning, and assessment.
- **Research artifacts** – experiment outputs and logs are stored in `Research_Runs/`.

## Reproducible Research
Consistent configuration files, deterministic seeds, and scripted training/evaluation workflows align the project with reproducible research practices, enabling others to replicate and extend the results.
