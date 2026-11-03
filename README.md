# Confidence-Adaptive Pruning for Heuristic-Guided Search

The repository now has parallel benchmark packages: `bench201/` owns the existing
six-decision NAS-Bench-201 experiments; `bench301/` adds a 22-decision DARTS domain.
Both use the same implementations in `algs/` and dispatch in
`search_registry.py`. Existing NB201 artifact paths/checkpoint formats remain
unchanged. Old root/module paths are transitional import/CLI shims.

For the new domain, open `bench301/runs/extract_bench301.py`,
`bench301/runs/train_bench301.py`, `bench301/runs/evaluate_bench301.py`, or
`bench301/runs/compare_bench301.py` and press Run after editing
the settings block. The four `bench301/runs/analyze_bench301_*.py` launchers perform
paired V-family and baseline studies with query budgets or guarded exhaustion.
Full workflow, commands, measured-data provenance, token
mapping and oracle runtime setup are in [bench301/README.md](bench301/README.md).
The full-log dataset has been extracted locally under
`bench301/artifacts/measurements.json`; these files are ignored by Git.

The requested change to absolute VCDFS local-bound interpolation was omitted:
the retention rule remains unchanged. The shared search core now accepts an
optional resource guard and optional batched domain scores; Bench201 does not
activate either. Old artifacts are preserved. The NB301 performance ensemble
and isolated XGBoost runtime are installed locally; no prefix transformer is
currently trained. The user runs that training from `bench301/runs/train_bench301.py`.

## Commands — run from the repository root

Your prompt must be inside the maintained clone containing this README and
`run.py`. On the current development machine:

```powershell
cd C:\Users\fires\Downloads\GURC-root\GURC-VCDFS-AdrienLawrence
```

If `run.py --help` lists only `inspect` and `demo`, you are in an obsolete copy.
The maintained runner lists `test`, `search`, `smoke`, `pilot`, and `full` too.

First-time environment setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### IDE Run-button launchers

The `bench201/runs/` directory contains one launcher per use case. Open a file, edit only
the uppercase settings block at its top, and press PyCharm/VS Code's Run button:

- `bench201/runs/extract_measurements.py` — build the compact NAS-Bench-201 measurement file;
- `bench201/runs/train_models.py` — train frozen transformer checkpoints;
- `bench201/runs/compare_algorithms.py` — run the baseline and algorithm comparison;
- `bench201/runs/analyze_v_family.py` — run the dedicated parameter-by-budget behavior suite;
- `bench201/runs/analyze_baselines.py` — run matched non-V baselines on the same models and budgets;
- `bench201/runs/analyze_v_family_exhaustive.py` — exhaust each retained absolute V-family tree;
- `bench201/runs/analyze_baselines_exhaustive.py` — exhaust each matched baseline tree;
- `bench201/runs/view_search_results.py` — open the baseline-comparison Pareto viewer;
- `bench201/runs/view_v_family_results.py` — explore V-family behavior by family and budget;
- `bench201/runs/view_baseline_results.py` — explore the matched baseline behavior artifact;
- `bench201/runs/view_search_tree.py` — generate the real-data root-to-leaf explanatory figure;
- `tests/run_tests.py` — run the complete correctness suite.

Each launcher uses the active Python interpreter and forces the repository root as
its working directory, so it does not depend on an IDE-specific working-directory
setting. The four behavior-analysis launchers extend a compatible existing artifact:
completed rows are skipped and newly configured algorithms are appended. All other
experiment outputs remain exclusive rather than being silently overwritten.

Routine commands:

```powershell
# Show command descriptions.
.\.venv\Scripts\python.exe -B run.py --help

# Run all correctness tests.
.\.venv\Scripts\python.exe -B run.py test

# Run a deliberately reduced, non-scientific one-query plumbing check.
.\.venv\Scripts\python.exe -B run.py smoke

# Run and save the complete 15-model, 100-query comparison.
.\.venv\Scripts\python.exe -B run.py search

# Run a compact held-out V-family behavior sweep across eight query budgets.
.\.venv\Scripts\python.exe -B run.py behavior --preset quick

# Run the exhaustive 90-configuration held-out V-family parameter sweep.
.\.venv\Scripts\python.exe -B run.py behavior --preset full --output artifacts\v-family-behavior-full-001

# Explore an aggregated V-family behavior result interactively.
.\.venv\Scripts\python.exe -B -m ui.behavior_cli --results artifacts\v-family-behavior-full-001

# Open the read-only Pareto UI after saving a search run.
.\.venv\Scripts\python.exe -B run.py ui --results artifacts\full-search-001

# Train split 101/model 0 for up to 500 epochs.
.\.venv\Scripts\python.exe -B run.py pilot

# Train the complete preset study: five splits times three model seeds.
.\.venv\Scripts\python.exe -B run.py full

# Inspect the benchmark archive metadata without training a model.
.\.venv\Scripts\python.exe -B run.py inspect

# Run the labeled synthetic search demonstration.
.\.venv\Scripts\python.exe -B run.py demo --trace

# Verify lint and formatting.
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

`pilot` writes `artifacts/pilot-study-next`; `full` writes
`artifacts/full-study`; `search` reads those 15 checkpoints and writes
`artifacts/full-search-001`. These commands refuse to overwrite existing output
directories. `search`, `pilot`, and `full` preserve their documented scientific
presets. Only `smoke` intentionally reduces the query budget.

`behavior` is a separate diagnostic suite. For every frozen checkpoint it searches
only that checkpoint's held-out NAS-Bench-201 test architectures. The quick preset
uses 18 representative V-family configurations; the full preset enumerates all 15
ordered bound pairs available at 20% proportional increments and all 15 integer
bound pairs from one through five, for VCDFS, corrected VCBFS, and VCBFSN. Both presets
run query budgets 1, 2, 4, 8, 16, 32, 64, and 100. The suite records optimal-path
survival, exact pruning events, final and anytime regret, query/expansion/score/time
to the final incumbent, retention, frontier pressure, and ordinary work counters.
It writes a recoverable `runs.partial.json` every ten runs, then atomically promotes
that checkpoint to `runs.json` after completion instead of storing both copies.
Rerunning either analysis launcher with the same output directory resumes after the
last saved row. Before continuing, the runner verifies the data checksum, preset,
budgets, selected checkpoints, algorithm parameters, and metric version. Changed
plans are rejected rather than combined. At most `CHECKPOINT_EVERY - 1` unsaved
runs are repeated after an abrupt interruption.
The held-out oracle is used only after each search
to label behavior; it never controls traversal or model fitting.
The console prints every run and the five strongest configurations at each budget,
ranked by anytime regret, final regret, and then heuristic scoring work.

The full sweep contains 10,800 raw runs, not 10,800 parameter configurations:
90 configurations × 8 budgets × 15 frozen models. `runs.json`/`runs.csv` preserve
those raw runs and event-derived traces. `split_summary.json` first averages the
three repeated model fits inside each split. `summary.json`/`summary.csv` contain
only 720 configuration-budget rows and are the normal analysis surface.

The third `absolute-ablation` preset isolates VCDFS-A, VCBFS-A, and VCBFSN-A. It evaluates
every ordered `(k_min, k_max)` pair from one through five: 15 configurations per
algorithm and 45 total. `--models-per-split` deterministically retains the lowest
model seeds in each split without discarding split coverage. The editable launcher
uses one model per split and budgets 1, 8, 32, 64, and 100, giving 45 × 5 × 5 =
1,125 runs.

Behavior rows now include clean runtime plus separately replayed Python-allocation
peak, whole-process RSS peak, and RSS increase. The separate deterministic replay
keeps memory instrumentation out of the runtime and pruning observations.

The separate `baselines` preset excludes the V family and reuses the same held-out
checkpoints, query budgets, metric definitions, and deterministic memory replay.
It contains DFS, BFS, heuristic DFS, GBFS, A*, MCTS-UCT, CDFS-1 through CDFS-5,
Beam-4/16, and the entropy and standard-deviation adaptive beams. The default
`bench201/runs/analyze_baselines.py` plan is
15 configurations × 5 budgets × 5 checkpoints = 375 matched runs. This avoids
duplicating V-family work already present in the ablation artifact.

The two `*_exhaustive.py` launchers impose neither expansion nor terminal-query
limits. Each algorithm stops only after exhausting the search tree that remains
reachable under its pruning policy. Exhaustive rows use `query_budget = -1` and
`budget_mode = "exhaustive"`; the UI labels this condition “Exhaustive.” Anytime
regret is still evaluated over the full held-out terminal count, carrying a
pruned algorithm's final incumbent forward so algorithms remain comparable.
Using one model per each of five splits limits this add-on to 225 V-family runs
and 60 baseline runs.

The behavioral viewer reduces this further to one family and one budget—normally
15 points. Its controls switch family, budget, work measure, and final versus
anytime regret. Horizontal position represents work, vertical position represents
regret, point color represents the rate at which every optimal held-out path was
pruned, marker size represents optimum-found rate, and the outlined connected
points form the selected Pareto frontier. The viewer is read-only and does not
save figures automatically.

### Root-to-leaf explanatory figure

Open `bench201/runs/view_search_tree.py`, edit its uppercase settings, and press Run. The
figure reruns the selected frozen model and V-family algorithm on that checkpoint's
held-out NAS-Bench-201 test partition. It automatically chooses a completed path
with a strong within-path confidence contrast and shows the lowest- and highest-
confidence internal decisions away from the leaf so every displayed branch remains
readable.

Every path node names the operation assigned to the current NAS edge, bolds the
next edge to be decided, reports the real frozen-transformer estimate `h(s)`, and
shows first-visit order. The terminal node gives the complete six-operation
architecture string, predicted error, measured NAS-Bench-201 validation error,
prediction error, and terminal-query position. Both decision panels list every
local child score and its retained/discarded status, the relevant confidence
population, top-two gap, mean, MAD, actual confidence equation, interpolated
`q(c)`, raw width, and actual retained width. For VCBFS, confidence statistics use
the pooled depth frontier while the displayed child decisions remain parent-local.
The caption records release, dataset, metric, epoch, held-out split seed, model
seed, controller bounds, epsilon, and algorithm. Like the other UI figures, it
opens interactively and never saves automatically.

The Pareto UI reads the versioned `summary.json`, computes nondominated points,
and opens an interactive Matplotlib window. Its selector switches among runtime,
RSS, queries, heuristic scores, expansions, and pending states. It does not save
or modify artifacts automatically; the Matplotlib toolbar remains available for
an explicit manual export.

Implementation repository for a 2026 Georgia Undergraduate Research Conference (GURC) project at the University of West Georgia.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the enforced one-way UI dependency
rule, immutable artifact contract, and versioned result/trace schema boundary.

This project investigates whether **confidence-adaptive pruning can reduce search cost faster than it increases optimization regret by selectively sacrificing completeness**.

The central idea is a pruning controller that combines two pieces of information from heuristic scores:

1. how strongly the best candidate is separated from the second-best candidate; and
2. how dispersed the broader candidate set is.

Rather than using heuristic dispersion alone to determine search width, the controller normalizes the top-two heuristic margin by the overall score dispersion, then maps that confidence value to a retained search width.

The controller is evaluated through two first-class search implementations:

- **VCDFS**: applies confidence-adaptive retention locally to each sibling set during depth-first traversal.
- **VCBFS**: computes one confidence value over the combined next-level scores,
  then applies the resulting retention independently within each parent's children.

The project studies whether changing retention bounds moves these methods along a useful empirical **regret-resource Pareto frontier**.

## Research Question

The project is not primarily asking whether VCDFS or VCBFS is universally more optimal than existing search algorithms.

Instead, it asks:

> Can confidence-adaptive pruning produce useful optimization quality-resource tradeoffs by reducing memory, runtime, and search work faster than solution quality degrades?

The algorithms intentionally sacrifice completeness when pruning occurs. Once a candidate is discarded, it is never revisited.

The relevant question is therefore not whether pruning preserves exhaustive search, but whether the loss of completeness buys enough computational savings to justify the resulting optimization regret.

## Confidence-Adaptive Retention

Assume lower heuristic values are better.

For a candidate set with sorted heuristic scores

$$
h_1 \le h_2 \le \cdots \le h_b,
$$

the controller computes the mean absolute deviation

$$
\operatorname{MAD}(H)
=
\frac{1}{b}
\sum_{i=1}^{b}
|h_i-\bar{h}|,
$$

and defines confidence as

$$
C(H)
=
\operatorname{clip}
\left(
\frac{h_2-h_1}
{\operatorname{MAD}(H)+\epsilon},
0,
1
\right).
$$

The numerator measures how strongly the best-ranked candidate separates from its nearest competitor.

The denominator provides context from the dispersion of the complete candidate set.

A high confidence value therefore means that the best candidate is **unusually better than the second-best candidate relative to the typical score deviation among the alternatives**.

The controller then interpolates between minimum and maximum retention bounds:

$$
k
=
Ck_{\min}
+
(1-C)k_{\max}.
$$

Higher confidence produces narrower search. Lower confidence retains more alternatives.

The resulting statistic is deterministic. It is **not** a probability that the best branch contains the optimum, a calibrated uncertainty estimate, or a statistical confidence interval.

### Intuition

Imagine choosing the largest Skittle from a pile.

If one Skittle is clearly larger than the next-largest Skittle while nearly every other Skittle is similarly small, the choice is relatively unambiguous.

If the entire pile contains wildly inconsistent sizes, a small advantage over the runner-up provides much weaker evidence that one candidate truly stands out.

The controller encodes the same intuition using heuristic scores: the top-two margin matters, but its meaning depends on the broader score distribution.

## VCDFS and VCBFS

VCDFS and VCBFS use the same confidence formulation but apply it to different candidate populations.

### VCDFS

VCDFS performs depth-first traversal.

At each expanded parent:

1. generate its feasible children;
2. score all children with the heuristic;
3. rank them from best to worst;
4. compute confidence from that sibling set;
5. determine retained width;
6. discard the remaining siblings;
7. explore retained children depth-first.

After reaching a terminal or dead end, VCDFS backtracks to unexplored retained siblings.

Pruned siblings are never reopened.

### VCBFS

VCBFS applies the same controller breadth-first.

At each depth:

1. expand the current frontier;
2. pool all child scores to compute one shared level-wide confidence;
3. preserve each parent's child group;
4. calculate a local width from that parent's branching count and the shared confidence;
5. retain the best children within every parent group;
6. concatenate the retained groups as the next frontier.

`VCBFSBeam` preserves the earlier pooled global-retention implementation as a
separately named adaptive-beam baseline. It may eliminate every child from a
weaker parent; corrected VCBFS cannot do so when that parent's local width is
positive.

### Proportional and Absolute Width Modes

Both algorithms support two ways of specifying retention bounds.

**Proportional mode** expresses minimum and maximum retention as percentages of the available candidates.

**Absolute mode**, exposed by the `-A` variants, expresses those bounds as literal node counts.

The `-A` forms do not define different confidence algorithms. They change only the units used to specify the retained-width interval.

## Optimization Setting

This study focuses on **terminal optimization over a finite construction tree**.

It is not conventional shortest-path search, first-goal search, or pathfinding.

Search continues after finding a terminal solution so that algorithms can improve the current incumbent until their retained search space or budget is exhausted.

The true optimization objective for an architecture \(a\) is

$$
E(a)
=
100
-
\operatorname{meanValidationAccuracy}(a),
$$

measured in percentage points.

Lower values are better.

Optimization regret is

$$
R
=
E_{\text{best found}}
-
E_{\text{global optimum}}.
$$

This is an absolute percentage-point difference, not a relative percentage.
The primary value is search-only: `search_best_error - global_oracle_error`.
The train/development incumbent is recorded separately as a diagnostic and does
not improve primary regret or algorithm ranking.

Because NAS-Bench-201 is fully enumerable, the global optimum under the selected validation measurement is known. This allows exact regret to be calculated rather than approximating solution quality from the best architecture encountered during search.

## NAS-Bench-201 Search Domain

Experiments use **NAS-Bench-201 v1.1**.

The search space contains:

- 6 sequential operation decisions;
- 5 possible operations per decision;
- \(5^6 = 15{,}625\) complete architectures;
- 19,531 total prefix states when the root, partial states, and terminals are included.

Each search state is a partial architecture assignment.

Because the six decisions are made in a fixed order, each complete architecture has one construction path. The resulting experimental domain is therefore a deterministic finite tree rather than an arbitrary cyclic graph.

The approximately 5 GB original NAS-Bench-201 archive is **not tracked in Git**. It must be obtained separately before rebuilding the extracted measurement dataset.

Generated measurements, checkpoints, training runs, and empirical artifacts are also excluded from version control.

## Learned Heuristic

NAS-Bench-201 provides validation measurements for completed architectures, but partial architectures do not have a naturally defined intermediate objective value.

To guide heuristic-based algorithms through the construction tree, this project trains an encoder-only transformer to estimate

$$
h(s)
\approx
\mathbb{E}
\left[
E(A)
\mid
A \text{ extends } s
\right].
$$

The target is the **mean terminal validation error among completions compatible with the partial architecture**.

It is not the minimum attainable descendant error.

The transformer therefore acts as a learned intermediate evaluation function derived from terminal outcomes.

It is experimental infrastructure, not the primary algorithmic contribution.

During search, trained weights are frozen and used as an external heuristic shared across competing heuristic-guided methods.

### Current Pilot Predictor

The current pilot transformer reports approximately:

- MAE: **2.14 percentage points**
- RMSE: **5.35 percentage points**
- Spearman correlation: **0.877**

The relatively strong rank correlation is especially relevant because the search algorithms primarily consume candidate ordering and score distributions rather than treating predicted errors as exact objective values.

These metrics come from a pilot model and should not be interpreted as final multi-split predictor results.

## Algorithms

The repository currently implements the following search methods.

### Traversal controls

- Depth-First Search
- Breadth-First Search

### Heuristic controls

- Heuristic-ordered DFS
- Greedy Best-First Search

### Fixed-retention methods

- Fixed-width Beam Search
- CDFS

CDFS is the fixed-width local predecessor/control for VCDFS. It ranks sibling states heuristically but retains a fixed number of candidates rather than adapting retention from confidence.

### Proposed adaptive family

- VCDFS
- VCBFS

Both use the normalized top-two-margin confidence controller described above.

### Literature adaptive-width baseline

- Dynamic Beam Search using Shannon entropy
- Dynamic Beam Search using top-\(k\) standard deviation

Dynamic Beam is the closest adaptive-width literature comparison in the current implementation because it also derives search width from properties of the candidate-score distribution.

The distinction is that Dynamic Beam uses entropy or standard deviation as the width-control statistic, while the proposed controller explicitly combines **ranking separation** with **overall dispersion**.

For NAS-Bench-201, predicted validation errors must first be adapted into a probability distribution before the Dynamic Beam policies can be applied.

## Evaluation

The primary scientific outcome is the relationship between **regret** and **resource use**.

The project asks whether different retention settings generate useful nondominated operating points rather than reducing performance to a single winner.

Resource measurements are prioritized as:

1. RSS memory increase
2. runtime
3. terminal objective queries
4. heuristic evaluations
5. node expansions
6. peak pending states

`peak_pending_states` is retained because it provides an implementation-independent structural measure of frontier or stack pressure even though it is not a byte-level memory measurement.

The empirical runner also records generated states, pruning counts, terminal counts, search depth, confidence statistics, stop reasons, and the best architecture found.

Runtime measurement is separated from memory instrumentation. Memory is measured using a deterministic replay, and the replay must reproduce the original search result and accounting before the measurement is accepted.

## Preliminary Results

The following results are from a **single data split and a single trained model initialization**.

They are preliminary observations only and do not establish statistical superiority.

| Method | Regret | Runtime | Expansions | Heuristic Scores | Terminal Queries | Peak Pending | RSS Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| VCBFSBeam-20-40% | 0.0000 pp | 0.0830 s | 17 | 85 | 11 | 40 | 20 KiB |
| CDFS-4 | 0.0000 pp | 0.1532 s | 37 | 185 | 100 | 19 | 24 KiB |
| Dynamic Beam StdDev 4-16 | 0.0000 pp | 0.1957 s | 55 | 275 | 16 | 80 | 28 KiB |
| GBFS | 0.0000 pp | 0.1385 s | 34 | 170 | 100 | 40 | 36 KiB |
| VCDFS-20-100% | 0.0000 pp | 0.3195 s | 73 | 365 | 100 | 12 | 40 KiB |
| VCDFS-A-1-2 | 0.3547 pp | 0.0256 s | 6 | 30 | 1 | 1 | 20 KiB |
| Beam-4 | 0.3547 pp | 0.0746 s | 21 | 105 | 4 | 20 | 24 KiB |

In the current pilot, the historical **VCBFSBeam-20-40** implementation found a
globally optimal architecture using 17 expansions, 85 heuristic evaluations,
and 11 terminal queries. These measurements must not be attributed to corrected
VCBFS; corrected VCBFS configurations require new runs.

The narrow VCDFS-A-1-2 configuration demonstrates the opposite end of the tradeoff: it used only 6 expansions, 30 heuristic evaluations, and one terminal query, but incurred 0.3547 percentage points of regret.

This is the behavior the project is intended to characterize.

Different retention bounds can move an adaptive method between very low-cost incomplete search and more expensive configurations capable of recovering the known optimum.

The complete empirical output contains additional methods and configurations beyond the condensed table shown here.

## Current Experimental Status

### Completed

- NAS-Bench-201 measurement extraction
- deterministic construction-tree representation
- transformer heuristic implementation
- pilot transformer training
- frozen checkpoint inference
- DFS and BFS controls
- heuristic DFS and GBFS
- fixed Beam Search
- CDFS
- VCDFS proportional and absolute-width modes
- VCBFS proportional and absolute-width modes
- entropy-based Dynamic Beam
- standard-deviation Dynamic Beam
- common search accounting and budget semantics
- runtime and memory instrumentation
- deterministic memory replay
- pilot search experiment
- unit and property testing

### In progress

- repeated transformer training
- final parameter validation
- Dynamic Beam development-set parameter selection
- expanded empirical comparisons
- Pareto-front analysis
- final tables and figures

### Planned Final Study

The planned replicated study uses:

- **5 architecture split seeds**
- **3 model initialization seeds per split**
- **15 trained transformers total**

Results will be aggregated in a split-aware manner.

Model initializations within the same architecture split are not treated as independent datasets.

The final study is intended to determine whether confidence-adaptive configurations consistently occupy useful regret-resource Pareto regions rather than relying on the favorable behavior of one pilot run.

## Repository Structure

```text
algs/
    Common search infrastructure and implementations for
    DFS, BFS, GBFS, Beam, Dynamic Beam, CDFS, VCDFS, and VCBFS.

heuristic/
    NAS measurement extraction, transformer model,
    training, evaluation, checkpointing, and experiment driver.

behavior/
    V-family diagnostic experiments and real search-trace capture.

ui/
    Read-only result viewers and figure renderers. UI code does not
    run algorithms or alter experiments.

bench201/runs/ and bench301/runs/
    Benchmark-specific editable IDE launch files. Both launch from the repository
    root so existing data and output paths remain unchanged.

schemas/
    Shared validation and artifact-writing infrastructure.

tests/
    Search correctness, property/adversarial testing,
    transformer, architecture-boundary, behavior, and runner tests. Run all via
    tests/run_tests.py.

nas_space.py
    NAS-Bench-201 parsing and deterministic six-decision
    construction-tree representation.

empirical.py
    Measures frozen-model search runs and coordinates result output.

search_specs.py
    Declares algorithm configurations and dispatches each algorithm.

search_reporting.py
    Calculates metrics and formats console, JSON, and CSV results.

tree_figure.py
    Captures a real NAS search trace, then sends immutable figure data
    to the read-only tree renderer.

run.py
    Stable command-line presets for tests, training, search, behavior,
    dataset inspection, and demonstrations.

PROTOCOL.md
    Formal experimental protocol, mathematical definitions,
    assumptions, and limitations.

IMPLEMENTATION.md
    File-by-file and function-level implementation reference.

requirements.txt
    Direct tested Python dependencies.

pyproject.toml
    Ruff lint and formatting configuration.

artifacts/
    Local generated measurements, checkpoints, and experiment
    outputs. Excluded from Git.
```

## Reproduction

The tested environment uses CPython 3.12 on CPU.

### Enter the repository and create the environment

Run every command below from the repository root:

```powershell
cd C:\\Users\\fires\\Downloads\\cs4275-gurc-research
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Simple preset commands

The top-level runner supplies complete presets so routine validation does not
depend on remembering long argument lists:

```powershell
.\.venv\Scripts\python.exe -B run.py test
.\.venv\Scripts\python.exe -B run.py smoke
.\.venv\Scripts\python.exe -B run.py search
.\.venv\Scripts\python.exe -B run.py pilot
.\.venv\Scripts\python.exe -B run.py full
```

`search` runs the full 100-terminal-query comparison against the existing
`pilot-study-long` checkpoint. `pilot` trains split 101/model 0 for up to 500
epochs into `artifacts/pilot-study-next`. `full` preserves the five-split by
three-model defaults and writes `artifacts/full-study`. Output directories are
never overwritten. Only `smoke` reduces the query budget to one, and it labels
itself non-scientific in the console.

### Extract NAS-Bench-201 measurements

Place the original `NAS-Bench-201-v1_1-096897.pth` archive in the repository root, then run:

```powershell
.\.venv\Scripts\python.exe -B -m heuristic.extract
```

The extraction process verifies the expected archive, reads the NAS measurements, and writes:

```text
artifacts/measurements.json
```

The original archive and generated measurement file are excluded from Git.

### Train one pilot transformer

```powershell
.\.venv\Scripts\python.exe -B -m heuristic.experiment `
  --output artifacts/my-pilot `
  --split-seeds 101 `
  --model-seeds 0
```

### Train the planned full study

```powershell
.\.venv\Scripts\python.exe -B -m heuristic.experiment `
  --output artifacts/my-study
```

The default full configuration uses five split seeds and three model initialization seeds.

### Run empirical search

```powershell
.\.venv\Scripts\python.exe -B empirical.py `
  --study artifacts/my-study `
  --output artifacts/search-study
```

For console-only execution without generated result files:

```powershell
.\.venv\Scripts\python.exe -B empirical.py `
  --study artifacts/my-study `
  --no-write-json `
  --no-write-csv
```

### Run tests

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Current status: **65 tests passing**.

Search-only tests, which do not require the transformer experiment to be rerun:

```powershell
.\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -p "test_search*.py" -v
```

### Lint

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

### Check formatting

```powershell
.\.venv\Scripts\python.exe -m ruff format --check .
```

To apply Ruff formatting:

```powershell
.\.venv\Scripts\python.exe -m ruff format .
```

## Reproducibility and Limitations

The current implementation assumes a finite deterministic construction tree rather than a general cyclic graph.

VCDFS and VCBFS are intentionally incomplete whenever candidates are pruned.

The confidence score is a ranking statistic, not model uncertainty.

NAS-Bench-201 objective queries are table lookups into previously measured architectures. Terminal-query counts therefore represent the number of architecture evaluations requested by search, not the wall-clock cost of training those architectures from scratch.

The current reported search results come from one architecture split and one transformer initialization.

They demonstrate that the implemented algorithms can produce different regret-resource operating points, but they do not establish general or statistically significant superiority.

Final claims will depend on the planned replicated study.

## Data and Generated Artifacts

The approximately 5 GB NAS-Bench-201 archive is intentionally excluded from this repository.

The following are also ignored:

- `.venv/`
- `*.pth`
- generated `artifacts/`
- Python caches
- Ruff caches

Raw checkpoints, training runs, and empirical outputs should remain local.

Only lightweight curated final tables or summaries needed to support published results are intended for eventual version control.

## References

The project builds on work including:

- Dong, X. and Yang, Y. **NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search.** ICLR, 2020.
- Merenda et al. **Sequence-To-Sequence Neural Networks Inference on Embedded Processors Using Dynamic Beam Search.** *Electronics*, 2020.
- Standard beam-search and heuristic-search literature used as methodological baselines and context.

References are included where they are directly relevant to implemented comparisons rather than to inflate the bibliography.

## License

Code in this repository is intended to be released under the **MIT License**.

The NAS-Bench-201 dataset/archive is not redistributed by this repository and remains subject to its own terms and licensing.

## Preparing a GitHub commit

Generated data and environments should remain absent from `git status`. Review the exact commit contents before publishing:

```powershell
git status --short
git add .
git status --short
git commit -m "Initial confidence-adaptive NAS search study"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Do not run `git remote add origin` if the remote already exists; inspect it with `git remote -v`. No dataset, checkpoint, or generated artifact should appear in the staged-file list.

## Author

**Adrien Lawrence**  
University of West Georgia  
GURC 2026 Undergraduate Research Project
