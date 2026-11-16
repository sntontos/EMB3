# EMB3

Benchmarking whole-graph embedding methods (Graph2Vec, FGSD, GIN) against a Weisfeiler–Lehman kernel baseline on ENZYMES, IMDB-MULTI and REDDIT-MULTI-12K. Evaluates classification, clustering and robustness to edge perturbations, plus accuracy vs. compute cost. NTUA Information Systems, Fall 2026-27.

## Setup

The only prerequisite is [uv](https://docs.astral.sh/uv/). It downloads the two Python interpreters and installs every package from the lock files.

```bash
scripts/setup_envs.sh
```

This builds two environments, because KarateClub has not been updated since 2022 and cannot share an environment with current PyTorch:

| Environment | Python | Used for | Lock file |
| --- | --- | --- | --- |
| `.venvs/karateclub` | 3.10.21 | Graph2Vec, FGSD (KarateClub 1.3.3) | `requirements/karateclub.txt` |
| `.venvs/main` | 3.12.14 | GIN (PyTorch Geometric), WL kernel (GraKeL), all evaluation | `requirements/main.txt` |

Run code in an environment through the wrapper. It makes the `emb3` package in `src/` importable and fixes Python's hash seed:

```bash
scripts/run.sh main path/to/script.py
```

### Docker

The same two environments on x86-64 Linux, built by the same setup script. The image is x86-64 only because the 2022 packages KarateClub needs have no ARM Linux wheels; on Apple Silicon Docker runs it under emulation.

```bash
docker build -t emb3 .
```

```bash
docker run --rm -it -v "$PWD/data:/emb3/data" -v "$PWD/results:/emb3/results" emb3
```

Timings from inside the container and from a native run are not comparable and must not be mixed.

## Data

```bash
scripts/prepare_data.sh
```

This downloads ENZYMES, IMDB-MULTI and REDDIT-MULTI-12K from TUDataset (about 60 MB), checks each archive against the SHA-256 in `configs/default.yaml`, and stores the graphs in one canonical form under `data/processed/`. Every method reads that form through a converter in `src/emb3/convert.py` (NetworkX for KarateClub, GraKeL, PyTorch Geometric), so all methods see identical graphs.

- **Node labels.** ENZYMES uses the node labels it ships with. IMDB-MULTI and REDDIT-MULTI-12K have none, so the node degree is the label. The 18 continuous node attributes of ENZYMES are not used, because only GIN could take them.
- **Splits.** `splits/<dataset>.json` holds the stratified folds (10 folds × 3 repeats for the small datasets, 5 folds × 1 for REDDIT-MULTI-12K) and, per fold, a stratified 10% validation set taken from the training graphs for model selection. The files are committed and never regenerated implicitly, so every method and every rerun uses the same partition.
- **Statistics.** `results/datasets/statistics.md` is the dataset table for the report. It matches the figures published by TUDataset.

## Reproducibility

- **Pinned versions.** `requirements/*.txt` are generated from the matching `.in` files with `uv pip compile`; the exact command is the first line of each lock file. The KarateClub lock is resolved as of 1 November 2022, which gives the library versions KarateClub 1.3.3 was released against.
- **Seeds.** All seeds are listed in `configs/default.yaml` and applied with `emb3.seeds.seed_everything`.
- **Protocol.** Folds, repeats, embedding dimensions, perturbation rates and timing repeats are in `configs/default.yaml`.
- **Hardware and software record.** `scripts/record_system.sh` writes `results/system/SYSTEM.md` and one JSON file per environment with every installed package version. Run it on the machine that produces the timings.

## Layout

```
configs/default.yaml   experiment protocol and seeds
requirements/          dependency inputs (.in) and pinned locks (.txt) for both environments
scripts/               setup, run wrapper, data preparation, system record
src/emb3/              shared Python package, importable from both environments
splits/                cross-validation splits (committed)
results/               tables, figures and the system record (committed)
data/                  downloaded datasets and cached embeddings (not committed)
```
