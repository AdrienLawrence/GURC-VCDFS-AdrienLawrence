# Architecture and dependency boundaries

## Benchmark ownership after the domain refactor

`bench201/` owns `space.py`, `specs.py`, `empirical.py`, the heuristic pipeline
and behavior suite. `bench301/` owns DARTS representation, a separate measured
data/training pipeline, terminal surrogate adapter and empirical runner.
`search_registry.py` owns neutral specifications and dispatch; presets belong to
each benchmark. There is one shared `algs/` package with no benchmark imports.

Legacy `nas_space.py`, `empirical.py`, `search_specs.py`, `heuristic/*` and
`behavior/*` are small import/CLI shims, not second implementations.
Existing launchers use canonical package paths and unchanged settings.
Saved NB201 artifacts/checkpoints remain in place with their original schemas.

`bench301/oracle_bridge.py` provides the isolated legacy process boundary.
Other NB301 imports do not require the surrogate environment. Download/extraction
are explicit; offline tests require no network or external binary.
NB301 result artifact kinds are separate so the NB201 regret viewer cannot
mistake surrogate objective values for known-optimum regret.

The repository separates experiment production from downstream presentation.
Search behavior and scientific measurements must never depend on visualization.

## Allowed dependency direction

```text
algs/ -----------+
heuristic/ ------+--> experiment producers --> artifacts/presentation data
nas_space.py ----+                                      |
behavior/ -------+                                      v
empirical.py ----+                              UI/readers/renderers
search_specs.py -+
search_reporting.py

run files / top-level entry points orchestrate producers and renderers

schemas/ <----------- imported by producers and consumers
```

`schemas/` is neutral infrastructure. It uses only the Python standard library
and defines versioned artifact envelopes that can be shared without coupling a
producer to a consumer.

The allowed rule is one-way:

- `ui/` may import `schemas/` and render saved artifacts or immutable presentation data.
- Experiment packages may import `schemas/` when writing artifacts.
- `algs/`, `heuristic/`, `behavior/`, `empirical.py`, `nas_space.py`, and `run.py` must never
  import `ui/`.
- `schemas/` must not import experiment or UI packages.

`tests/test_architecture.py` enforces these import restrictions by parsing the
repository source. A forbidden core-to-UI import fails the test suite.

## Read-only UI contract

The UI is a renderer, not an experiment controller. It may:

- select and load an existing manifest, result, or trace;
- filter, inspect, and visualize values already present in that artifact;
- render or export figures derived from a loaded artifact.
- render immutable presentation data captured by an upstream producer.

It must not:

- modify checkpoints, configurations, benchmark data, or saved experiment files;
- import live algorithm objects and advance or alter their state;
- recompute controller decisions and substitute them for recorded decisions;
- write into raw experiment directories;
- influence seeds, budgets, parameter selection, or algorithm execution.

Loaded artifacts are recursively frozen by `schemas.ArtifactEnvelope`. UI state
such as the selected node or zoom level must remain separate from scientific
artifact data. Figure exports are new downstream products and must retain source
provenance; they are not mutations of their input artifacts.

## Versioned artifact envelope

New result and trace artifacts use exactly three top-level fields:

```json
{
  "schema_version": 1,
  "artifact_kind": "run_result",
  "payload": {}
}
```

Schema versions are positive integers scoped to an artifact kind. Unknown kinds,
stale versions, non-finite JSON numbers, malformed required fields, and undeclared
top-level extensions are rejected rather than guessed. Consumers must fail closed
when an artifact version is unsupported.

New heuristic-training and empirical-search JSON outputs use this envelope.
Extracted measurement files and transformer checkpoints retain their existing
independently versioned formats (`schema: 1` and `schema: 2`, respectively).
The empirical runner accepts old unwrapped `split.json` files only through an
explicit legacy-read path so existing frozen pilot checkpoints remain usable;
newly generated splits are always enveloped.
