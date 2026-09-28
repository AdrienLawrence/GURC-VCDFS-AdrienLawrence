# Frozen prefix-transformer experiment, version 2

This is the implemented protocol, approved for implementation on September 22,
2026. `revised_paper.txt` is the earlier proposal, not an exact specification of
the current experiment. Changes to this protocol require a new version; results
from different versions must not be silently combined.

## Research question and boundaries

With one shared frozen learned heuristic, how does confidence-adaptive local
pruning change the quality/work trade-off relative to fixed retention and plain
traversal? This is an optimization experiment, not a completeness demonstration.
The transformer is infrastructure, not a claim of a novel state-of-the-art NAS predictor.
The six-level, five-way NAS-Bench-201 construction tree is a small controlled
benchmark: 15,625 complete encodings and 19,531 total states. It cannot establish
generality to deep production search or all architecture spaces.

## Objective and data identity

- Original `NAS-Bench-201-v1_1-096897.pth`, MD5
  `55e847143ce1f7c2d89b676f6b096897`; not the newer NATS-Bench release.
- `cifar10-valid`, `x-valid@199`, 200-epoch training regime (`full`).
- Error is `100 - mean(available recorded validation accuracies)` in percentage
  points. Keep every trial seed and accuracy in the extracted measurement file.
- Trial coverage is uneven: 7,961 architectures have seeds 777/888/999; 5,439
  have 777/888; 2,225 have only 888. Averaging does not remove this limitation.
  A common-seed-888 study is a possible separately declared sensitivity analysis,
  not a change made opportunistically after examining algorithm rankings.
- No CIFAR test score is exported, used for fitting, or used to select a model.
- Uniform means uniform over the benchmark's 15,625 operation encodings. Distinct
  encodings may express equivalent effective networks; we do not claim an
  equivalence-class-disjoint test or deduplicate/change the search space here.

## What is predicted

For prefix s, the desired value is the mean validation error of uniformly chosen
complete architectures extending s. It is neither the minimum reachable error
nor a probability of success nor an admissible lower bound on the minimum.

Sample complete architectures uniformly without replacement. For every training
architecture a with observed error E(a), create six examples (prefix_d(a), E(a)),
for d = 1..6. There is no root example: algorithms rank nonempty child states.
Every depth has equal total training weight. Shared prefixes can have several
different outcome labels; that is deliberate, not a contradictory dataset.

Minimize L = (1/(6N)) sum_a sum_d (f(prefix_d(a)) - E(a))^2.
For a fixed input, squared loss is minimized by the conditional mean. Finite
samples, shared network parameters, regularization and imperfect optimization
mean the fitted transformer need not attain that population target. Weighted aggregation
of duplicate prefixes is equivalent; UNWEIGHTED prefix means are not.

Only observed training completions supply labels. No exact full-subtree mean,
minimum, oracle ranking, or held-out outcome enters fitting. No recursive
descendant inspection occurs at prediction time.

## Tokenization and transformer

Native edge order: 0->1, 0->2, 1->2, 0->3, 1->3, 2->3. A prefix is represented
as six integer tokens. IDs 0..4 are the five NAS operations; ID 5 is UNASSIGNED.
UNASSIGNED is not the `none` operation. The learned model receives categorical
tokens, not a flat 36-value feature vector.

The encoder-only transformer regressor works as follows:

1. Map every operation token to a learned 32-value operation embedding.
2. Add a learned 32-value embedding identifying its fixed edge position.
3. Prepend one learned SUMMARY token. Mask unassigned suffix positions from
   attention; the summary token and assigned operation positions remain visible.
4. Apply two pre-layer-normalized transformer encoder blocks. Each uses four
   attention heads, model dimension 32, a 64-unit GELU feed-forward sublayer,
   residual connections, and dropout 0.10.
5. Normalize the summary representation and linearly regress one scalar.

There is no decoder, text generation, causal mask, or language vocabulary.
Bidirectional attention is appropriate because all assigned operations in the
prefix are known simultaneously. Parameter count is 17,633. This compact size is
an engineering choice for a six-token problem, not an optimality claim.

Normalize target errors by TRAIN-only mean and population standard deviation;
use scale 1 when variance is effectively zero. Undo normalization before search.
Do not clip predictions to [0,100]: clipping alters VCDFS's score gaps. Instead
report out-of-range predictions. Reject nonfinite inputs, outputs and weights.

## Fitting and reproducibility

- 1,024 training architectures; 256 development architectures; 14,345 remaining
  architectures held out from fitting and checkpoint selection.
- Split architectures BEFORE creating prefixes. Shared prefixes across different
  architectures are legitimate; shared architecture outcomes are not.
- Seeds for splits: 101, 203, 307, 409, 503. Model seeds: 0, 1, 2.
- CPU, one PyTorch thread, deterministic algorithms. Reproducibility is tested
  within this runtime; bitwise equivalence across all hardware/releases is not
  promised. Versions and source/data hashes are saved with each experiment.
- AdamW, learning rate .001, weight decay .0001, batch 128; shuffled examples.
- Maximum 300 epochs. An epoch visits all 6,144 training examples once. Stop after
  30 epochs without a STRICT decrease in development MSE. Restore the best epoch.
- No hyperparameter tuning from held-out or search results. Different seeds do
  not change these hyperparameters. Development labels influence checkpoint
  choice and therefore count toward the initial 1,280 known outcomes.
- Checkpoints are non-executable JSON weights plus normalization and provenance.
  The predictor holds no benchmark labels. Gradients are disabled during search.
- There is no online learning, uncertainty ensemble, or best-completion target.
  Three initializations are separate trials, not an ensemble prediction.

## Diagnostics

Held-out residual MAE/RMSE and Spearman rank correlation are reported separately
at each depth. At shallow depths, error against individual completed outcomes
includes unavoidable within-prefix outcome variation; it is NOT direct error
against exact conditional means. Report top-decile terminal diagnostics too:
strong global ranking may merely distinguish broken networks from usable ones.
The top-decile threshold is evaluated post-fit, never used for checkpoint choice.

Prediction baselines, using identical training observations:
1. Constant training mean.
2. Observed prefix mean; unseen prefixes fall back to their longest observed
   ancestor, eventually the root mean.
3. Linear ridge regression over the same encoding, penalty 1, unpenalized bias.

After the model is frozen, a separate diagnostic uses ALL benchmark outcomes to
calculate exact prefix means/minima. This is clearly labeled POSTHOC ORACLE, not
held-out generalization evidence. It measures equal-prefix mean prediction by
depth, sibling pair ordering, best-mean-child choice, best-minimum-child choice,
and parent/mean-child prediction inconsistency. These tables never guide search.
In particular, correct mean ranking does not guarantee preservation of a branch
containing the global minimum.

## Search protocol

`empirical.py` executes this section only against already-frozen checkpoints.
It never trains or updates the heuristic during search.

Full construction tree; do not prune out training/development architectures or
build a held-out-only trie. Every algorithm gets the same initial best objective
among the 1,280 known architectures, in addition to any newly found incumbent.
The implementation combines this incumbent at reporting time: these algorithms
have no incumbent-based pruning, so this does not change their traversal.

100 terminal evaluations per run by default. Every terminal visit counts,
including already-known architectures. Report unique newly exposed architecture
labels separately; these are not claimed to be equal across runs. Counting
revisits is conservative and explicit, not an equal-new-query experiment.
All terminal solutions are evaluated with the same objective; no first-goal stop.
Fewer queries after retained-subtree exhaustion is a legitimate result.

Default empirical registry: DFS, BFS, DFS-H, GBFS, CDFS-2/4, Beam-4/16,
DynamicBeam-Entropy-4-16, DynamicBeam-StdDev-4-16, VCDFS percentages
20-40 / 20-80 / 20-100, VCDFS-A bounds 1-2 / 1-3 / 2-5, and VCBFS percentages
20-40 / 20-80. These settings are mechanism probes, not claimed to be optimally
tuned or equivalent effective widths. The two Dynamic Beam defaults use fixed,
declared NAS-adapter settings: entropy has slope 4 and intercept 0; top-5
standard deviation has slope -100 and intercept 16; both use temperature 1 and
clamp width to 4--16. Development-tuned alternatives must enter through an
external recorded specification. In b=5, integer flooring frequently collapses
narrow ranges.

Record expansions, generated/scored states, terminal queries, pruned states,
maximum depth, queue/stack high-water entries, width histograms, confidence
saturation, and wall-clock time. `peak_pending` is NOT bytes or peak RAM. The
runner additionally reports peak traced Python allocation bytes from a separate
deterministic replay. That value excludes PyTorch-native storage and model
weights, and is never mislabeled as total process RAM. Sampled process RSS and
its increase over the post-model-load baseline are also reported; RSS includes
the interpreter, model, native libraries, and allocator behavior.
No prediction cache is used. Timing is CPU wall time, not simulated CNN training
cost. Training time is reported separately. A mean-of-trials objective uses more
than one underlying CNN training run for some architectures: 1,280 labels does
not mean 1,280 CNN training jobs. We make no end-to-end GPU-cost claim.

Regret = best observed validation error - full-table minimum validation error,
in percentage points, not relative percent and not CIFAR test error.
Oracle minimum is calculated for reporting after searches finish.
Compare paired runs; three model seeds sharing a split are not three independent
datasets. Report variability; do not treat 15 runs as 15 independent tasks.
These runs are initial mechanism checks, not a final competitive NAS benchmark:
MCTS, evolutionary NAS, VCBFSN and online adaptation are not implemented in this
standalone version. GBFS and VCBFS are implemented. DFS/BFS are traversal
controls, not strong NAS claims. The default Dynamic Beam settings are explicitly
labeled mechanism probes, not tuned competitive configurations; tuned settings
require development-only selection through a recorded external specification.

## Exact algorithm semantics and limits

All algorithms require immutable states in a finite construction TREE, stable
successor enumeration, pure feasibility and a frozen deterministic heuristic.
They have no graph cycle detection or transposition dominance. Infeasible
children are filtered before scoring. Stable ties preserve successor order.

DFS keeps pending siblings on a stack and resumes them after terminals/dead ends.
CDFS/VCDFS score ALL feasible siblings before selecting the retained set. They
exhaust retained branches unless a budget interrupts them. Discarded branches
are not reconsidered. Thus retained-tree exhaustion is not global completeness.

Confidence = clip((h2-h1)/(mean absolute deviation of all sibling h + 1e-5),0,1).
For b=0, keep zero; b=1, keep one without confidence. Ties produce zero confidence.
Width = floor(conf*lower + (1-conf)*upper), multiplying by b/100 in proportional
mode, then clamping to [1,b] for nonempty sets. Absolute bounds are integers.
Clipped confidence is the September assignment convention, not a claim that all
versions of the original manuscript agree. No statistical confidence guarantee.

Beam collects every generated candidate at the next level, then keeps the global
best W. Terminals can be pruned before visitation. BFS never calls the heuristic.
GBFS keeps all pending states in one heuristic-priority queue and prunes none.
VCBFS scores the full next-level candidate set, calculates the top-two-gap/MAD
confidence over that level, and retains a confidence-dependent percentage of the
globally ranked candidates. Its plateau confidence is 0.5 and its width uses
half-up rounding, matching the reference implementation.
Dynamic Beam Search is included as two literature baselines: entropy-controlled
and top-k-standard-deviation-controlled global beam width. The source method uses
decoder probabilities. For NAS, predicted errors are converted with
softmax(-error/temperature); temperature and line parameters must be tuned using
development architectures only and reported. This is an adaptation, not a claim
of reproducing the paper's translation and summarization experiments.
Query limit stops immediately upon reaching the limit. Expansion limit stops
at the next nonterminal that needs expansion; it does not scan past that node
for later terminals. Pending terminals reached before that point can be evaluated.

DFS-family pending storage is O(1+dK+B) state records, not simply O(d), including
temporary sibling storage; state payloads and model/dataset memory are extra.
Each expanded node pays for generating/scoring B children and sorting them,
O(B*C_h+B log B), even when only one is retained. No claim of cost-free pruning.

## Research precedents (not prescriptions for our hyperparameters)

- Merenda et al., Sequence-To-Sequence Neural Networks Inference on Embedded
  Processors Using Dynamic Beam Search, Electronics 2020, 9(2), 337:
  https://doi.org/10.3390/electronics9020337
- Wen et al., Neural Predictor for Neural Architecture Search, ECCV 2020:
  https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740647.pdf
- White et al., BANANAS, AAAI 2021: https://arxiv.org/abs/1910.11858
- White et al., How Powerful Are Performance Predictors in Neural Architecture
  Search?, NeurIPS 2021: https://arxiv.org/abs/2104.01177
- Dong and Yang, NAS-Bench-201, ICLR 2020: https://arxiv.org/abs/2001.00326
- Official serialization/metric code:
  https://github.com/D-X-Y/NAS-Bench-201/tree/master/nas_201_api
