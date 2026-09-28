# Implementation review: what was added and why

This is the review guide, not another implementation. Functions are kept short
and code is formatted. Nothing trains implicitly on import. All algorithms
minimize objective values; higher accuracy is converted to lower error once.

## `algs/`: what each algorithm does

### `common.py` — rules everyone must follow

- `Problem` lists the domain's six operations: root, successors, feasibility,
  terminal test, heuristic and terminal objective. It contains no domain logic.
- `finite` rejects NaN/infinity; `positive_integer` rejects invalid widths.
- `Budget` validates optional expansion/query caps; it is immutable.
- `Result` stores the incumbent, stop reason and work counters.
- `Run.visit` checks feasibility, evaluates a terminal, or generates feasible
  children. It enforces the same budgets/counters across every scheduler.
- `Run.rank` scores all supplied children and stably sorts by score only.
- `Run.emit` invokes an optional diagnostic callback. Exceptions are not hidden.
- `Run.pressure` records queued entry count, not bytes or total RAM.

### `dfs.py` — the backtracking loop

- `dfs` pops the stack, visits the state, optionally selects children, and pushes
  retained children in reverse order so the best is visited next. Other siblings
  stay on the stack. Reaching a terminal does NOT finish the whole run.
- `heuristic_dfs` orders all children but discards none. Plain `dfs` does not
  invoke the heuristic. A selector cannot return an out-of-range width.

### `cdfs.py` — fixed local retention

- `cdfs` validates k and supplies `min(k,b)` as the selector to DFS. Sharing DFS
  avoids duplicated backtracking code; this file explicitly defines its rule.

### `vcdfs.py` — adaptive local retention

- `confidence` computes the best/second-best gap divided by mean absolute
  deviation plus epsilon, clipped to [0,1]. It includes all feasible siblings.
- `Retention` validates limits and computes the integer retained count. `all`
  and `fixed` modes also support the small experiment dispatcher.
- `vcdfs` constructs the proportional selector; `vcdfsa` constructs the absolute
  selector. Both use the same DFS backtracking. No hidden restart or reopening.

### `bfs.py`, `gbfs.py`, `beam.py`, `dynamic_beam.py`, and `vcbfs.py`

- `bfs` uses a FIFO queue, without scoring or pruning.
- `beam` explicitly constructs the next level, globally sorts its candidates,
  and keeps W. This is not CDFS performed breadth-first. Earlier terminal levels
  are evaluated on visitation; beam-pruned terminal candidates are not evaluated.
- `gbfs` retains every generated state in one global priority queue and always
  expands the state with the smallest heuristic. It does not use path cost and
  does not prune; its incompleteness in experiments comes only from budgets.
- `vcbfs` constructs the complete next-level candidate set, computes the
  best/second-best gap divided by level MAD, and maps that confidence into a
  proportional global frontier width. `vcbfsa` supplies absolute bounds. The
  reference plateau confidence is 0.5 and width rounding is half-up.
- `dynamic_beam` implements both controllers from Merenda et al. (2020): Shannon
  entropy over the full distribution and standard deviation over its top-k values.
  Both linearly map their statistic to a bounded global beam width. Because NAS
  heuristics are costs rather than decoder probabilities, `costs_to_probabilities`
  exposes a temperature-scaled softmax adapter instead of hiding that mismatch.

Standard beam fixes one width for every level. Entropy Dynamic Beam uses all
converted probabilities and widens when Shannon uncertainty is high. Standard-
deviation Dynamic Beam uses only the top-k probabilities and narrows when their
spread is high. VCBFS uses no entropy, top-k setting, temperature, slope, or
intercept: it directly maps the MAD-normalized top-two gap to the requested
minimum/maximum retained fraction. All four remain global breadth schedulers.

### `__init__.py` — public imports

Exports named algorithms and a small `search` dispatcher for existing settings.
It routes to the separate files; it no longer contains their traversal loops.
No VCBFS/N, MCTS, GBFS or graph-search guarantees are implied by the package name.

## `nas_space.py` and `run.py`

Existing NAS parsing/encoding/tree code retained, with formatting only.
`read_metadata` reads architecture strings without loading measurements.
`parse_architecture` and `architecture_string` are inverse conversions.
`encode` remains a legacy one-hot utility for transparent linear baselines.
`ArchitectureTree` supplies the five possible next operations and passes the
partial operation-ID tuple directly to the injected predictor.
The optional allowed-architecture trie remains available but the experiment
uses the full tree. `run.py` only changed imports/formatting; its demo is synthetic.

## `empirical.py` — frozen-model search measurements

- Discovers every `split-*-model-*` checkpoint produced by the training driver.
- Verifies checkpoint/data checksums and complete disjoint split files.
- Reconstructs the full 15,625-architecture tree and never updates the model.
- Runs every registered algorithm with the same terminal-query budget.
- Times an ordinary run, then performs a deterministic replay under `tracemalloc`
  so memory instrumentation does not contaminate the runtime measurement.
- Records solution quality, regret, expansions, generated/scored states, queries,
  pruning, depth, frontier pressure, Python allocation peak, sampled process RSS,
  width histograms and controller diagnostics.
- Writes per-run and split-aware JSON/CSV. Model seeds are averaged within split
  before final means/standard deviations are calculated across independent splits.
- Accepts external JSON algorithm specifications. This is required for tuned
  Dynamic Beam parameters; arbitrary untuned literature settings are not hidden
  in the default comparison.

## `heuristic/`: transformer learning and evaluation

### `extract.py` — create a small measurement table

- `extract` validates the official file MD5, processes records, checks 15,625
  unique IDs/encodings, and writes JSON without overwriting an existing file.
- `StreamingReader` is a release-specific restricted pickle interpreter. Only
  OrderedDict, NumPy dtype/scalar and codecs.encode constructors are permitted.
  Each completed record is summarized and large histories are discarded.
- `CompactMemo` stores pickle reference IDs compactly, preserving shared scalar
  values and small immutable tuples. References to discarded containers fail
  explicitly. This avoided loading a 5 GB object graph into limited RAM.
- `summarize` takes ONLY final 200-epoch CIFAR validation scores, retains trial
  seeds, and computes available-trial mean error. Test scores are not exported.
- This is not a generic untrusted-pickle sandbox. The official checksum is a
  required precondition. Files of any other release are rejected.

### `data.py` — keep partitions honest

- `load_measurements` checks schema, complete coverage, unique encodings, finite
  valid scores, trial uniqueness, and arithmetic agreement between trials/mean.
- `split_indices` uses a private seeded RNG to split architecture IDs BEFORE any
  prefix examples exist. `select_records` selects that partition explicitly.
- `examples` produces six prefix/outcome pairs per complete architecture; it
  rejects duplicates and invalid labels. No descendant oracle is consulted.

### `model.py` — the actual transformer

- `PrefixTransformer` embeds six categorical edge tokens and their positions in
  32 dimensions, prepends a learned summary token, and applies two four-head
  encoder blocks with 64-unit GELU feed-forward sublayers.
- Unassigned suffix tokens are padding-masked. The summary token reads assigned
  positions through bidirectional self-attention and is regressed to one scalar.
  There is no decoder or text generation. Total parameters: 17,633.
- `FrozenPredictor` puts the network in evaluation mode, disables gradients,
  converts predictions back to error percentage points, and rejects nonfinite
  values. `predict` is batch inference; `__call__` handles one search state.
- `save`/`load` use JSON, not executable pickle. Architecture, shapes and weights
  are checked. The saved predictor contains no benchmark outcome table.

### `train.py` — supervised fitting, no search

- `TrainingConfig` contains the declared defaults and rejects invalid settings.
- `fit` receives TRAIN and DEV records only. It rejects architecture overlap,
  seeds CPU training, builds prefix examples, and normalizes from TRAIN only.
- Each minibatch: predict, calculate mean squared residual, backpropagate,
  update weights with AdamW. Each epoch: measure train/development MSE.
- Keep the checkpoint with strictly lowest development MSE; stop after 30
  non-improving epochs or 300 total. Restore that checkpoint and freeze it.
- Returns predictor and audit history. It cannot inspect held-out outcomes
  because they are not passed into this function. No online adaptation.

### `evaluate.py` — distinguish predictions from evidence

- `metrics` computes MAE, RMSE and tie-aware Spearman correlation; undefined
  correlation is JSON null, not a fabricated zero.
- `evaluate` reports held-out results separately by depth, top-decile terminal
  results and out-of-range prediction counts.
- `Baselines` fits constant, observed-prefix fallback and ridge-linear predictors
  using TRAIN only. `baseline_evaluation` applies the same depth metrics.
- `oracle_diagnostics` explicitly reads ALL outcomes after fitting and measures
  exact subtree-mean error, sibling choices and parent/child consistency. These
  are diagnostic oracle results, never training labels or search inputs.

### `experiment.py` — reproducible heuristic orchestration

- `main` loads measurements, creates disjoint splits, trains each declared seed,
  freezes models, and evaluates the heuristic. It never runs search algorithms.
- `source_digest` fingerprints algorithm/model/domain source. Manifests also
  record measurement hash, Python/dependency versions, seeds and settings.
- `save_json` refuses overwrite. Partial runs retain completed subdirectories;
  rerun into a new output folder rather than silently mixing experiments.

## Tests: what the suites establish

- `tests/test_search.py`: the original 20 tests moved intact except imports and
  formatting. Covers confidence arithmetic, degeneracy, ties, budgets, exact
  visit order, pruning failure, deep traversal, beam scope and NAS encoding.
- `tests/test_search_properties.py`: independent recursive reference on 30
  random finite trees; public entry-point equivalence; blind traversal never
  scoring; terminals never expanding; invalid scores/widths; query caps.
- `tests/test_transformer.py`: disjoint deterministic splits, token handling,
  duplicate-prefix weighting,
  label validation, lookup fallback, tie-aware metrics, exact network size,
  frozen checkpoint roundtrip, rejected overlap/nonfinite weights, deterministic
  fitting, train-only normalization, ability to learn a simple synthetic mapping,
  and restricted extraction compared with ordinary pickle on synthetic records.

Synthetic tests are correctness checks, NOT NAS performance evidence. Passing
them cannot guarantee absence of bugs, nor prove completeness of a pruning rule.

## Configuration and documents

- `requirements.txt`: tested direct dependency versions; installed in `.venv`.
- `pyproject.toml`: formatting/lint rules. Frozen dataclass defaults are marked
  immutable; loop-closure lint exceptions are limited to synchronous callbacks.
- `.gitignore`: excludes dataset, environment, cache and generated artifacts.
- `README.md`: runnable commands and entry points.
- `PROTOCOL.md`: current research specification, definitions and caveats.
- `revised_paper.txt`: earlier draft preserved, now labeled historical.

## Recoverability

The refactor replaces the two root-level implementation/test files with packages.
Dataset contents are never edited. No Git commit, external publication, algorithm
benchmark, or additional domain has been created.
