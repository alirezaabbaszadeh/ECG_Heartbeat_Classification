# Reproducibility guide

This guide separates two different goals:

1. **Artifact reproducibility**: deterministically verify the exact prediction
   arrays and summary values reported in the manuscript.
2. **Computational replication**: download the public source data and train new
   models from the source code.

Artifact verification is fast and is the appropriate first check. Full training
is hardware-intensive and may not be bit-for-bit identical because the original
environment was not captured as a complete package lockfile and some GPU
operations can be nondeterministic.

## Fixed study design

| Item | Fixed value |
| --- | --- |
| Database | MIT-BIH Arrhythmia Database v1.0.0 |
| Included records | 45 |
| Excluded records | 102, 104, 107 |
| Development records | 38 |
| Held-out test records | 228, 207, 208, 233, 220, 231, 106 |
| Random seed | 42 |
| ECG channel | First channel |
| Beat window | 187 samples: 93 before, centre, 93 after |
| Wavelet | Morlet (`morl`) |
| Scales | Integers 1-32 |
| Sequence length | Three consecutive beats |
| Prediction target | Third beat |
| Classes | Normal, SVEB, VEB, Fusion, Unknown/paced |
| Final epochs | 20 |
| Final batch size | 128 |
| Optimiser | AdamW, learning rate 1e-5 |
| Loss | Sparse categorical cross-entropy with balanced class weights |

The canonical split is stored independently for every architecture. The four
`data_splits.json` files are byte-identical and are included in the checksum
manifest.

## Level 1: verify the fixed article outputs

Only NumPy is needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-verify.txt
python3 verify_article_artifacts.py
```

The verifier performs all of the following:

- checks SHA-256 for 36 configurations and output artifacts;
- checks that the four architectures use the same label vector;
- checks the seven-record test split;
- confirms that stored predicted classes equal the probability argmax;
- recomputes accuracy, macro-F1, weighted-F1, and five one-vs-rest AUC values;
- verifies test support of 12,061 N, 214 S, 2,913 V, 383 F, and 2 Q targets;
- verifies the CNN-Conformer five-fold mean accuracy of 0.441546 and standard
  deviation of 0.161235.

The expected values and file hashes are versioned in
[`article_artifacts.json`](article_artifacts.json).

## Level 2: regenerate the complete pipeline

### 1. Prepare the environment

The pinned reference stack targets 64-bit Linux or WSL and Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

The archived `.keras` files report Keras 3.8.0. The training machine was
recorded as Ubuntu under WSL, Python 3.12.10, NVIDIA GeForce GTX 1660 Ti
(6 GB), Intel Core i7, and 16 GB RAM. No complete historical `pip freeze` or
container digest was retained. The reference requirements are therefore
explicit and installable, but should not be interpreted as proof of every
historical transitive dependency.

### 2. Download the source database

```bash
python3 download_data.py
```

This creates `mit-bih-arrhythmia-database-1.0.0/`. The data are downloaded
from PhysioNet and are not stored in this repository.

### 3. Generate unnormalised scalograms

```bash
python3 preprocess_data.py
```

This writes one HDF5 file per included record under
`preprocessed_data_h5_raw/`. Normalisation is deliberately deferred. During
model development, means and scales are fitted using the relevant training
records only.

### 4. Build three-beat TFRecords

```bash
python3 create_batched_tfrecords.py
```

This writes the derived sequence data under `tfrecord_data_batched/`.

### 5. Run an individual architecture

Valid internal architecture names are `Main_Model`, `AttentionOnly`,
`CNNLSTM_Model`, and `Baseline_Model`.

```bash
python3 run_hyperparameter_tuning.py --model_name Main_Model
python3 run_kfold_evaluation.py --model_name Main_Model
python3 run_final_evaluation.py --model_name Main_Model --epochs 20 --batch_size 128
```

The final script loads the latest matching tuning configuration,
hyperparameters, and data split from `Research_Runs/`. Keep unrelated newer
tuning directories out of that folder when attempting to follow the archived
configuration exactly.

### 6. Run all four architectures

```bash
bash run_full_pipeline.sh
```

New outputs use timestamped directories and do not overwrite the fixed article
artifacts.

## Mapping from article names to artifacts

| Article name | Internal name | Fixed final-run directory |
| --- | --- | --- |
| CNN-Conformer | `Main_Model` | `final_run_Main_Model_20250824_154136` |
| CNN-attention | `AttentionOnly` | `final_run_AttentionOnly_20250824_090531` |
| CNN-LSTM | `CNNLSTM_Model` | `final_run_CNNLSTM_Model_20250824_213453` |
| CNN-only | `Baseline_Model` | `final_run_Baseline_Model_20250824_182728` |

For each final run:

- `raw_predictions.npz` contains `y_true`, `y_pred_probs`, and
  `y_pred_classes`;
- `classification_report.txt` contains the rounded text report;
- `final_model_*.keras` contains the trained model;
- `final_model_training_history.json` contains the epoch history;
- the PNG files contain diagnostic plots.

The CNN-Conformer development-fold values are in
`Research_Runs/kfold_eval_Main_Model_20250823_204942/kfold_summary.json`.

## Interpretation and known limitations

- The fixed numerical record is the archived prediction arrays, not a promise
  that a new GPU training run will generate identical weights.
- The test set contains only two Q targets, so its class-specific AUC is not
  treated as inferential evidence in the article.
- Beat-level bootstrap intervals in the manuscript do not account for
  within-record clustering or uncertainty in selecting a new patient cohort.
- The models were evaluated on one historical database and are not cleared or
  validated for clinical use.
- Raw or derived waveform data must be handled under the PhysioNet database
  licence and should not be committed to this repository.
