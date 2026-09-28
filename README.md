# Confidence-Adaptive Pruning for Heuristic-Guided Search

Implementation repository for a 2026 Georgia Undergraduate Research Conference (GURC) project at the University of West Georgia.

This project investigates whether **confidence-adaptive pruning can reduce search cost faster than it increases optimization regret by selectively sacrificing completeness**.

The central idea is a pruning controller that combines two pieces of information from heuristic scores:

1. how strongly the best candidate is separated from the second-best candidate; and
2. how dispersed the broader candidate set is.

Rather than using heuristic dispersion alone to determine search width, the controller normalizes the top-two heuristic margin by the overall score dispersion, then maps that confidence value to a retained search width.

The controller is evaluated through two first-class search implementations:

- **VCDFS**: applies confidence-adaptive retention locally to each sibling set during depth-first traversal.
- **VCBFS**: applies the same confidence controller globally to the combined candidate frontier at each search depth.

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
2. combine all generated candidates for the next level;
3. score and rank that global candidate set;
4. compute one confidence value across the level;
5. determine retained width;
6. retain the best candidates as the next frontier.

VCBFS therefore tests whether the confidence formulation remains useful when applied globally rather than to local sibling decisions.

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
| VCBFS-20-40% | 0.0000 pp | 0.0830 s | 17 | 85 | 11 | 40 | 20 KiB |
| CDFS-4 | 0.0000 pp | 0.1532 s | 37 | 185 | 100 | 19 | 24 KiB |
| Dynamic Beam StdDev 4-16 | 0.0000 pp | 0.1957 s | 55 | 275 | 16 | 80 | 28 KiB |
| GBFS | 0.0000 pp | 0.1385 s | 34 | 170 | 100 | 40 | 36 KiB |
| VCDFS-20-100% | 0.0000 pp | 0.3195 s | 73 | 365 | 100 | 12 | 40 KiB |
| VCDFS-A-1-2 | 0.3547 pp | 0.0256 s | 6 | 30 | 1 | 1 | 20 KiB |
| Beam-4 | 0.3547 pp | 0.0746 s | 21 | 105 | 4 | 20 | 24 KiB |

In the current pilot, **VCBFS-20-40 found a globally optimal architecture using 17 expansions, 85 heuristic evaluations, and 11 terminal queries**.

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

tests/
    Search correctness, property/adversarial testing,
    transformer tests, and empirical-runner tests.

nas_space.py
    NAS-Bench-201 parsing and deterministic six-decision
    construction-tree representation.

empirical.py
    Runs frozen-model search experiments and produces
    per-run and split-aware empirical summaries.

run.py
    Dataset inspection and synthetic search demonstrations.

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
