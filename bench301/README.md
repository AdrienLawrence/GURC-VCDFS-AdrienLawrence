# NAS-Bench-301 integration

The search implementations in `algs/` are shared with NAS-Bench-201. Their
traversal and retention rules are unchanged. An optional external stop callback
and batch scoring hook support Bench301; Bench201 does not activate either.
The prefix predictor and the terminal surrogate are separate frozen models.
No existing artifact is migrated, rewritten, or deleted by this refactor.

## Run-button workflow

Open a file in `bench301/runs/`, edit its uppercase settings, then press Run. For
training, run `bench301/runs/train_bench301.py`; running the implementation module
`bench301/heuristic/experiment.py` directly loses Python package context.

- `bench301/runs/extract_bench301.py`: official result archive -> grouped measured labels.
- `bench301/runs/train_bench301.py`: measured labels -> frozen prefix checkpoints.
- `bench301/runs/evaluate_bench301.py`: print held-out real-data model diagnostics.
- `bench301/runs/compare_bench301.py`: shared algorithms + prefix checkpoint + terminal oracle.
- `bench301/runs/create_bench301_reference.py`: optional frozen random quality reference.
- `bench301/runs/analyze_bench301_v_family.py`: V-family behavior at 1, 4 and 8 queries.
- `bench301/runs/analyze_bench301_v_family_exhaustive.py`: guarded, unbudgeted V-family search.
- `bench301/runs/analyze_bench301_baselines.py`: matched baselines at 1, 4 and 8 queries.
- `bench301/runs/analyze_bench301_baselines_exhaustive.py`: guarded, unbudgeted baselines.

Extraction has already produced `bench301/artifacts/measurements.json` locally
from the official full-log archive. The launcher refuses to overwrite it.
Choose a different output if intentionally extracting another source.

The former small pilot was pipeline validation, not a main experimental result,
and no pilot artifact is retained. The training launcher now targets
`bench301/artifacts/full-study`, uses five split seeds and one model seed per
split, the complete 80/10/10 data split, and 300 epochs. Training is still an
explicit Run-button action and is never started by import, tests, or setup.

The four behavior launchers share that five-checkpoint study and use fresh,
non-overlapping output directories. Budgeted launchers use terminal-query
budgets 1, 4 and 8. Exhaustive-mode launchers still pass explicit time, RSS-delta
and expansion safety limits; a safety stop is recorded rather than silently
treated as completed exhaustion. Edit the uppercase constants before Run when a
new artifact identifier or different resource envelope is required.

## Verified official sources

- [Official repository and download links](https://github.com/automl/nasbench301).
- [Measured dataset v1.0](https://figshare.com/articles/dataset/NAS-Bench-301_Dataset_v1_0/13246952).
- [Full training logs](https://figshare.com/articles/dataset/nasbench301_full_data/13286105).
- [Official ConfigLoader](https://github.com/automl/nasbench301/blob/master/nasbench301/surrogate_models/utils.py).
- [Genotype/ConfigSpace conversion](https://github.com/automl/nasbench301/blob/master/nasbench301/representations.py).
- [XGBoost training target](https://github.com/automl/nasbench301/blob/master/nasbench301/surrogate_models/gradient_boosting/xgboost.py).
- [Deterministic API](https://github.com/automl/nasbench301/blob/master/nasbench301/api.py).
- [Legacy dependencies](https://github.com/automl/nasbench301/blob/master/requirements.txt).

The published archives were downloaded explicitly and verified against their
published MD5 values. Downloads/imports/tests never happen implicitly.

## Representation

Each cell contributes 11 sequential decisions:

| Destination | Decisions | Branching |
|---|---|---|
| 2 | operation from source 0, operation from source 1 | 7, 7 |
| 3 | unordered predecessor pair, lower-source op, higher-source op | 3, 7, 7 |
| 4 | unordered predecessor pair, lower-source op, higher-source op | 6, 7, 7 |
| 5 | unordered predecessor pair, lower-source op, higher-source op | 10, 7, 7 |

Normal cell comes first, reduction cell second. Total depth is 22.
The forced predecessor pair at node 2 has no separate search decision.

Operation tokens 0..6, in order:
`max_pool_3x3`, `avg_pool_3x3`, `skip_connect`, `sep_conv_3x3`,
`sep_conv_5x5`, `dil_conv_3x3`, `dil_conv_5x5`.

Pair tokens 7..16 follow lexicographic `combinations(range(5), 2)`.
Token 17 means UNASSIGNED. Pair tokens are legal only at positions whose
destination has those predecessors. Sorting edges preserves operation/source
association. Both concat lists are exactly `[2, 3, 4, 5]`.

The space contains
`(7**8 * 3 * 6 * 10)**2 = 1,076,746,950,455,072,400`
canonical complete architectures.

## Measured labels and leakage controls

The exact official raw architecture field is
`optimized_hyperparamater_config` (including its misspelling).
The target is `info[0]["val_accuracy"]` in percent; error is `100 - accuracy`.
This matches the official loader and XGBoost target.

These records have a 100-epoch training budget, 32 initial channels, eight
layers, 40,000 training images and 10,000 validation images. The reported
`epochs` identifies the selected summary. The summary usually differs from
`val_accuracy_final`; extraction preserves both and never calls the selected
summary a final-epoch measurement. `test_accuracy` and learning-curve test
labels are never exported as targets.

Complete architectures are canonicalized and grouped before splitting. Every
source observation and its checksum are retained. Labels average available
validation observations. Source records do not expose training seeds; `seed`
is explicitly null rather than guessed from a filename. Multiple observations
are not described as independently seeded trials without such evidence.

Real verification on 2026-10-14:

| Source | Observations | Unique canonical architectures | Repeated observations |
|---|---:|---:|---:|
| Compact measured dataset | 59,328 | 53,558 | 5,770 |
| Full training logs | 57,189 | 51,418 | 5,771 |

The archives are distinct datasets; their counts/checksums must not be substituted.
The current local measurements use the full-log archive.

The architecture train/dev/test split is independent for the new prefix
transformer. The pretrained official terminal surrogate was developed from
historical NAS-Bench-301 measurements and is not retrained for each of our
splits. Its online predictions must therefore be described as surrogate
search quality, not independent measured validation outcomes.

Split complete canonical architectures first, then generate depths 1..22.
Each measured architecture contributes equally at each depth. The lazy dataset
stores complete states once and constructs padded prefixes on demand.

MSE learns the empirical mean error conditional on a prefix in the measured
sampling distribution. The released data include optimizer-selected samples:
this is not proof of uniform expectation over the entire DARTS space.

## Prefix transformer

The encoder uses dimension 32, four heads, two layers, FF dimension 64, GELU,
dropout 0.10, a learned summary token and a scalar output.
Inputs are `[batch, 22]` integer tensors. Embeddings sum value, position,
decision type (operation/pair) and cell type (normal/reduction).
UNASSIGNED suffix positions are attention masked.

AdamW updates weights from training batches only. Development MSE selects and
restores the best checkpoint; early stopping is deterministic.
Normalization comes only from training errors. Held-out records enter evaluation
after fitting. JSON checkpoints include benchmark, encoding, token map,
architecture settings, normalization, split/model seeds, source checksum and
offline training counts. The NB201 and NB301 loaders reject one another's formats.

Held-out reporting includes MAE, RMSE and Spearman at each depth, plus terminal
best-decile diagnostics. Optional `prefix_mean_diagnostics` compares estimates
with empirical held-out descendant means. Optional `surrogate_alignment`
compares prefixes with Monte Carlo completion means; it is not training data.

## Terminal oracle and runtime isolation

Only complete states are converted to DARTS genotypes and passed to official
`load_ensemble`/`predict(..., representation="genotype", with_noise=False)`.
The v1.0 XGBoost target is validation accuracy percent; online objective is
`100 - predicted validation accuracy`.

The adapter caches complete-state predictions. Caches are cleared for each
algorithm and its memory replay so prior algorithms do not supply free objective
results to later ones. Terminal queries are logical complete-state objective
evaluations; `surrogate_calls` separately counts cache misses.

Upstream `api -> utils` eagerly imports GNN/BANANAS/LightGBM/sklearn machinery
even for XGBoost. Its dependencies include torch 1.5 and compiled geometric
extensions. Those dependencies are absent in the current Python 3.12 environment.
No dependency was installed or downgraded.

Supply an independently prepared NB301 runtime through `NB301_PYTHON` or
`--oracle-python`. The main process communicates with a persistent JSONL
worker. The official backend uses `oracle_bridge.py`; the installed minimal
XGBoost backend uses `xgb_worker.py` with the verified official encoding and
model weights. Worker startup is outside search timing; query IPC is inside
timing. Startup and prediction have finite timeouts.
Memory metrics cover the main process, excluding the worker's allocations/RSS.

`requirements-nb301.txt` documents this boundary rather than issuing an unverified
legacy install recipe. The pretrained 1.85 GB model archive and a compatible
oracle interpreter are required for actual surrogate searches. The installed
minimal XGBoost runtime works with `ORACLE_BACKEND = "minimal-xgb"` and avoids
importing the unneeded GNN/BANANAS/LightGBM stack. A mocked oracle validates
encoding, noise-disabled calls, caching and metrics without the model.

## Retention conflict resolved in favor of the latest constraint

The ticket's Phase 3 requested clipping absolute bounds BEFORE interpolation.
That would change existing algorithm behavior when `kmax > b`, and the latest
user constraint explicitly forbids it. The retention rule remains unchanged.

`VCDFS-A(2,10)` interpolates global bounds and THEN clips its width to local
branching. It must not be described as interpolating between 2 and local `b`.
All NB301 results record the existing rule. A future change requires a separately
approved algorithm revision and separately identified results.

## Search outputs

NB301 records observed surrogate error, complete state/genotype/config, work
counters, stop reason, timing, memory, width/confidence distributions and the
best-so-far improvement curve. Milestones identify query/expansion/score/time to
the final best result.

There is no known global optimum, so NB301 has no global regret, optimum-found
rate, optimal-path survival or fatal-optimum-pruning metric.
A frozen random reference optionally supplies `reference_gap_pp` and a sample
percentile. Reference gap may be negative. Neither quantity proves global
optimality. Offline acquisition/training counts are documented separately from
online terminal queries and prefix-inference counts.

The single-comparison control file still contains VCDFS-A(2,10), CDFS-2,
CDFS-3, heuristic DFS and Beam-4. The behavior studies declare 20 V-family
settings and 13 baselines. Query budgets are 1, 4 and 8. Every algorithm uses
the same frozen predictor on a given split and the same deterministic oracle.
Default per-run guards are 180 seconds, 512 MiB additional main-process RSS,
and 5,000 expansions. A slow terminal query can overshoot a cooperative guard.
Only `stop=exhausted` denotes natural exhaustion; resource-limited runs are
censored. No-terminal runs remain in the data with `found=false` and null
quality.

Each behavior study writes `manifest.json`, resumable `runs.partial.json`,
completed `runs.json`/`runs.csv`, `split_summary.json`/`split_summary.csv`,
and `summary.json`/`summary.csv`. The best-so-far curve and mean best error over
a fixed terminal-query horizon provide anytime measurements. Models are
averaged within split before the five split means are compared. These splits
test sensitivity to the learned heuristic, not five independent NAS spaces.

## Module commands

Run from the maintained repository root with the existing Python environment:

```powershell
# Download full measured logs or the compact measured dataset explicitly.
.\.venv\Scripts\python.exe -B -m bench301.download full-data
.\.venv\Scripts\python.exe -B -m bench301.download data

# Download pretrained surrogate archive explicitly; approximately 1.85 GB.
.\.venv\Scripts\python.exe -B -m bench301.download models
Expand-Archive -LiteralPath bench301\cache\nasbench301_models_v1.0.zip -DestinationPath bench301\cache\models

# Extract full real measurements. Existing output is never overwritten.
.\.venv\Scripts\python.exe -B -m bench301.heuristic.extract --source bench301/cache/nasbench301_full_data.zip --output bench301/artifacts/measurements-next.json

# Train the main five-split, one-model-per-split study on the full 80/10/10 partition.
.\.venv\Scripts\python.exe -B -m bench301.heuristic.experiment --data bench301/artifacts/measurements.json --output bench301/artifacts/full-study --split-seeds 101 203 307 409 503 --model-seeds 0 --epochs 300

# Evaluate the frozen prefix predictor on held-out REAL records.
.\.venv\Scripts\python.exe -B -m bench301.heuristic.evaluate --data bench301/artifacts/measurements.json --checkpoint bench301/artifacts/full-study/split-101-model-0

# Point to a separately installed official NB301 interpreter.
$env:NB301_PYTHON = "C:\path\to\independent-nb301-env\python.exe"

# Replace model path with the extracted xgb_v1.0 ensemble directory.
.\.venv\Scripts\python.exe -B -m bench301.empirical --checkpoint bench301/artifacts/full-study/split-101-model-0 --oracle-model bench301/cache/nb_models_1.0/xgb_v1.0 --queries 1 --oracle-python bench301/cache/oracle-runtime/python.exe --oracle-backend minimal-xgb --no-write-json

# Optional frozen random reference.
.\.venv\Scripts\python.exe -B -m bench301.reference --oracle-model bench301/cache/models/nb_models_1.0/xgb_v1.0 --size 1000 --seed 101 --output bench301/artifacts/reference-001.json

# Full offline correctness and style checks.
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```
