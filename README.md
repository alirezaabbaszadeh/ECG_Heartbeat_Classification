# Record-Level Evaluation of a Morlet CNN-Conformer for ECG Beat Classification

This is the official code-and-artifact repository for the manuscript:

> **Record-Level Evaluation of a Morlet CNN-Conformer for Five-Class ECG Beat Classification**

It contains the source pipeline, fixed record split, selected hyperparameters,
trained Keras models, and prediction arrays used for the reported analyses. The
stable submission snapshot is
[`v1.0-joe-submission`](https://github.com/alirezaabbaszadeh/ECG_Heartbeat_Classification/releases/tag/v1.0-joe-submission).

## Scope

The study uses 45 records from the
[MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/).
Records 102, 104, and 107 are excluded. Seven complete records are held outside
model development:

`228, 207, 208, 233, 220, 231, 106`

The other 38 records form the development set. Each target is the third beat in
a sequence of three consecutive 32 x 187 Morlet scalograms derived from the
first ECG channel. The five labels are Normal, SVEB, VEB, Fusion, and
Unknown/paced.

No PhysioNet waveform or annotation file is redistributed here. The repository
stores code and derived model outputs only.

## Reported held-out results

All four models were evaluated on the same 15,573 target beats.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| CNN-Conformer | 0.597 | 0.263 | 0.677 |
| CNN-attention | 0.269 | 0.149 | 0.379 |
| CNN-LSTM | 0.179 | 0.154 | 0.286 |
| CNN-only | 0.136 | 0.069 | 0.226 |

The CNN-Conformer performed best on aggregate metrics, but it did not provide
dependable five-class recognition: its F1 was 0.002 for SVEB and 0.007 for
Fusion. The repository should therefore be treated as a research artifact, not
as a clinical diagnostic system.

## Verify the published artifacts

This check does not retrain a model. It verifies SHA-256 digests, confirms that
all four models use identical held-out labels, and recomputes accuracy, F1, and
one-vs-rest AUC directly from the archived prediction arrays.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-verify.txt
python3 verify_article_artifacts.py
```

Expected final line:

```text
PASS: 36 checksums, four common test arrays, four metric sets, and the main-model cross-validation summary were verified.
```

The machine-readable specification is
[`article_artifacts.json`](article_artifacts.json).

## Re-run the complete experiment

The full workflow is computationally expensive:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
bash run_full_pipeline.sh
```

The script downloads the source database from PhysioNet, creates unnormalised
HDF5 scalograms, packages three-beat sequences as TFRecords, and then performs
hyperparameter search, five-fold development evaluation, final training, and
held-out evaluation for all four architectures.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact stages, artifact
mapping, recorded environment information, and limitations on bit-for-bit
training reproducibility.

## Repository map

| Path | Purpose |
| --- | --- |
| `download_data.py` | Download MIT-BIH v1.0.0 from PhysioNet |
| `preprocess_data.py` | Extract 187-sample beats and calculate Morlet scalograms |
| `create_batched_tfrecords.py` | Build three-beat TFRecord sequences |
| `DataLoader.py` | Construct record-based TensorFlow datasets and apply training-only normalisation |
| `ModelBuilder.py` | Define CNN-Conformer, CNN-attention, CNN-LSTM, and CNN-only models |
| `run_hyperparameter_tuning.py` | Select architecture-specific hyperparameters |
| `run_kfold_evaluation.py` | Evaluate development folds |
| `run_final_evaluation.py` | Fit on 38 development records and evaluate seven held-out records |
| `Research_Runs/` | Fixed configurations, histories, trained models, reports, plots, and prediction arrays |
| `article_artifacts.json` | Verified file digests and expected article metrics |
| `verify_article_artifacts.py` | Deterministic artifact and metric checker |

## Environment provenance

The training host was recorded as Ubuntu under WSL with Python 3.12.10, an
NVIDIA GeForce GTX 1660 Ti, and 16 GB RAM. The saved Keras files report Keras
3.8.0. A complete historical `pip freeze` was not retained; consequently,
[`requirements.txt`](requirements.txt) is a pinned reference stack compatible
with the archived models, not a claim that every transitive package is the
exact historical build. The fixed prediction arrays and checksums are the
authoritative record of the reported numerical results.

## Data and licensing

The MIT-BIH Arrhythmia Database is distributed by PhysioNet under the
[Open Data Commons Attribution License v1.0](https://physionet.org/content/mitdb/view-license/1.0.0/).
Users must download it from the source and cite the database:

G. B. Moody and R. G. Mark, “The impact of the MIT-BIH Arrhythmia Database,”
*IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45-50, 2001.
[doi:10.1109/51.932724](https://doi.org/10.1109/51.932724)

Repository code is available under the [MIT License](LICENSE).

## Citation

GitHub can generate a citation from [`CITATION.cff`](CITATION.cff). Until the
article receives its own bibliographic record, cite the fixed software release
and include its version or archive DOI.

## Authors

1. **Alireza Abbaszadeh** — Department of Computer Engineering, Islamic Azad
   University, Mashhad, Iran; ORCID
   [0009-0007-8253-6042](https://orcid.org/0009-0007-8253-6042);
   [alireza.abbaszadeh8558@iau.ir](mailto:alireza.abbaszadeh8558@iau.ir)
2. **Vahid Torkzadeh** — Department of Computer Engineering, Ma.C., Islamic
   Azad University, Mashhad, Iran
