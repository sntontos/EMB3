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

## Methods

| Method | Library | Kind | Environment | Output |
| --- | --- | --- | --- | --- |
| Graph2Vec | KarateClub 1.3.3 | unsupervised, fitted on all graphs of a dataset | `karateclub` | one vector per graph |
| FGSD | KarateClub 1.3.3 | unsupervised, each graph on its own, no training | `karateclub` | one vector per graph |
| GIN | PyTorch Geometric | supervised, trained per cross-validation fold | `main` | class probabilities and one vector per graph |
| WL subtree kernel | GraKeL | unsupervised kernel, the baseline | `main` | graph-by-graph kernel matrix |

Each method has a command that computes and stores one result; the experiment stages below call them. Results go to `data/embeddings/` and `data/gin/`. Next to each result, a `.json` file records the parameters, wall time, CPU time, peak memory, thread counts and library versions.

### Configuration that differs from the library defaults

- **Embedding dimension.** Graph2Vec: `dimensions`. FGSD: the number of histogram bins. GIN: the hidden width, because the pooled hidden vector is the graph embedding.
- **Graph2Vec.** `attributed=True`, so the node labels of the dataset (or the node degree) are used, and `workers=1`, because gensim is only repeatable with one worker. KarateClub's default of 10 epochs is kept as the setting `library`; `epochs100` trains for 100 epochs.
- **FGSD.** KarateClub's histogram range (0, 20) is kept. FGSD ignores node labels by design. On a few large REDDIT-MULTI-12K graphs KarateClub's pseudo-inverse fails ("SVD did not converge"); for those graphs only, the same pseudo-inverse is computed from the eigendecomposition, and their positions are stored in the job record.
- **GIN.** The network follows PyTorch Geometric's `mutag_gin.py` example: 5 GINConv layers with batch normalisation, sum pooling, an MLP classifier with dropout 0.5, Adam. The learning rate is 0.001 instead of the example's 0.01, with which the training loss on ENZYMES stayed at chance level. Node labels enter as one-hot vectors; on REDDIT-MULTI-12K, where degrees reach 3062, the degree enters as one standardised number (PyTorch Geometric's benchmark suite uses the same rule: one-hot below a maximum degree of 1000).
- **WL kernel.** GraKeL's `WeisfeilerLehman` with the `VertexHistogram` base kernel, normalised, with 1, 3 or 5 iterations.
- **Threads.** Every library is limited to one thread (`threads` in `configs/default.yaml`), so all methods get the same resources. Graph2Vec, the WL kernel and the SVM are single-threaded in any case.
- **KarateClub adds a self-loop to every node** of the graphs it is given. This is the library's behaviour and is left as it is; it affects Graph2Vec and FGSD only.

### How cost is measured

`emb3.measure.Meter` wraps each stage (input conversion, training, embedding generation) and records wall time, CPU time and the peak resident memory of the process, sampled every 20 ms and checked against the operating system's own high-water mark. Memory includes the interpreter, the libraries and the loaded dataset; the value before the stage is stored as well, so the stage's own share can be read off. Every job runs in a fresh process.

## Running the experiments

```bash
scripts/run_experiments.sh all
```

runs every stage, first for ENZYMES and IMDB-MULTI (a few hours), then for REDDIT-MULTI-12K (about 20 hours). Each stage can also be run on its own, and finished jobs are skipped, so an interrupted run can simply be started again. A failed job is listed in `logs/failed_jobs.txt` and the run goes on.

| Stage | What it does | How it runs |
| --- | --- | --- |
| `embeddings` | Graph2Vec and FGSD embeddings at 32, 128 and 512 dimensions, WL kernels; each repeated 5 times | timed, one job at a time |
| `gin-timed` | The GIN folds whose training time is reported (5 per dataset and dimension) | timed, one job at a time |
| `gin` | All remaining GIN folds of all candidate settings | 2 at a time (`JOBS=2`), low priority |
| `classify` | Model selection and test metrics per dataset, method and dimension | one job at a time |
| `summarize` | Result tables and the accuracy-versus-cost figure | |

A stage can be restricted, for example `scripts/run_experiments.sh embeddings --datasets ENZYMES IMDB-MULTI`.

To run it in the background and keep a log:

```bash
nohup scripts/run_experiments.sh all > logs/experiments.log 2>&1 &
```

```bash
scripts/progress.sh
```

shows what is done, what is running and the estimated time left.

**Keeping the laptop comfortable.** The timed stages use one CPU core, which is also what makes their timings comparable; only the untimed `gin` stage uses two, at low priority. The script keeps the Mac awake but lets the display sleep. Before every job that runs alone it checks the power source and waits while the Mac is on battery, so all timed runs happen on mains power. Every job record stores the power source, Low Power Mode and any thermal limit at the time it ran. During the timed stages, light use of the Mac (browsing, editing) is fine; heavy work (video calls, builds, games) would distort the timings. The lid must stay open: a closed lid puts the Mac to sleep.

### Classification protocol

- **Splits.** Stratified 10-fold cross-validation repeated 3 times on ENZYMES and IMDB-MULTI; one 5-fold run on REDDIT-MULTI-12K. Repeat *r* uses the embeddings of seed *r*.
- **Model selection.** Each method has four candidate settings, listed under `classification` in `configs/default.yaml`. In every fold the candidate with the best accuracy on that fold's validation graphs is chosen; the test fold plays no part. On REDDIT-MULTI-12K the first candidate is used unchanged (fixed hyperparameters).
- **Same data for every method.** All classifiers are fitted on the fold's training graphs and scored on its test graphs. Nothing is refitted on training plus validation, because GIN needs the validation graphs to pick its epoch.
- **Classifiers.** Graph2Vec and FGSD: an RBF SVM on the standardised embedding. WL kernel: an SVM on the precomputed kernel. GIN: trained end to end.
- **Metrics.** Accuracy, macro-F1 and macro one-vs-rest AUC on each test fold, in `results/classification/`. Tables give the mean and the standard deviation across test folds.

### Timing protocol

- **Clean runs only.** A timing counts only if the job ran alone, in a fresh process, with one thread. The run script marks those jobs; anything else is ignored by the cost tables.
- **What is measured.** Training and embedding generation of each method's reference setting. Library imports, loading the dataset and converting it to the library's format are excluded and stored separately.
- **Cold and warm.** Every run is a cold start: a new process, no warm-up run discarded.
- **Repeats.** 5 runs per dataset, method and dimension. Graph2Vec: seeds 0 to 4. FGSD and the WL kernel have no randomness and are simply run 5 times. GIN: 5 cross-validation folds.
- **Reported.** The median, the mean with its standard deviation and the 95% confidence interval of the mean (Student t), and the peak memory, in `results/cost/`.

## Reproducibility

- **Pinned versions.** `requirements/*.txt` are generated from the matching `.in` files with `uv pip compile`; the exact command is the first line of each lock file. The KarateClub lock is resolved as of 1 November 2022, which gives the library versions KarateClub 1.3.3 was released against.
- **Seeds.** All seeds are listed in `configs/default.yaml` and applied with `emb3.seeds.seed_everything`.
- **Protocol.** Folds, repeats, embedding dimensions, perturbation rates and timing repeats are in `configs/default.yaml`.
- **Hardware and software record.** `scripts/record_system.sh` writes `results/system/SYSTEM.md` and one JSON file per environment with every installed package version. Run it on the machine that produces the timings.

## Layout

```
configs/default.yaml   experiment protocol and seeds
requirements/          dependency inputs (.in) and pinned locks (.txt) for both environments
scripts/               setup, run wrapper, data preparation, experiment stages, system record
src/emb3/              shared Python package, importable from both environments
splits/                cross-validation splits (committed)
results/               tables, figures and the system record (committed)
data/                  downloaded datasets and cached embeddings (not committed)
```
