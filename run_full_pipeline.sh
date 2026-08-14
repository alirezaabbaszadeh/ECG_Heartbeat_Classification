#!/usr/bin/env bash
set -euo pipefail

# Regenerate the data representation and run all four architectures reported
# in the article. This is computationally expensive and requires the complete
# training environment described in REPRODUCIBILITY.md.

python3 download_data.py
python3 preprocess_data.py
python3 create_batched_tfrecords.py

for model in Main_Model AttentionOnly CNNLSTM_Model Baseline_Model; do
  python3 run_hyperparameter_tuning.py --model_name "$model"
  python3 run_kfold_evaluation.py --model_name "$model"
  python3 run_final_evaluation.py --model_name "$model"
done
