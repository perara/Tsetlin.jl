# Tsetlin.jl Benchmarks

This directory contains a dependency-free benchmark suite for the Fuzzy-Pattern Tsetlin Machine implementation and related utilities.

The runner is intentionally plain Julia so benchmark runs do not require a package environment:

```shell
julia -O3 -t auto benchmarks/runbenchmarks.jl --quick
```

## Modes

- `--quick`: smoke-sized coverage for development and pre-commit checks.
- `--full`: broader local comparison suite with more repeats and larger cases.
- `--exhaustive`: wide grid over vector sizes, literal densities, sparse-index behavior, multiclass cases, and TEXT HDC workloads.

Examples:

```shell
julia -O3 -t auto benchmarks/runbenchmarks.jl --quick
julia -O3 -t auto benchmarks/runbenchmarks.jl --full --format=csv --output=benchmarks/results/full.csv
julia -O3 -t auto benchmarks/runbenchmarks.jl --exhaustive --format=json --output=benchmarks/results/exhaustive.json
```

## Coverage

The suite measures:

- `TMInput` construction, indexing, mutation, summation, and `booleanize`.
- Clause primitives: `check_clause`, sparse literal index updates, and literal counting.
- Model operations: `vote`, `predict`, `train!`, batch prediction, `accuracy`, `compile`, `save`, `load`, `literals_count`, and `literals_sum`.
- Explainability helpers in `src/utils/explain.jl`.
- `fastconv` in 1D, 2D, and 3D cases.
- TEXT HDC helpers: random hypervectors, n-grams, and context hypervector generation.

`TS.benchmark` has its own output-heavy behavior, so it is opt-in:

```shell
julia -O3 -t auto benchmarks/runbenchmarks.jl --full --include-built-in
```

## Output

Supported formats are `markdown`, `csv`, and `json`.

```shell
julia -O3 -t auto benchmarks/runbenchmarks.jl --full --format=markdown --output=benchmarks/results/latest.md
```

Benchmark result files are machine-specific. Prefer keeping them out of commits unless you are deliberately publishing a reference run.

## Notes

Run with `-O3` and a fixed thread count when comparing two code versions. For example:

```shell
julia -O3 -t 16 benchmarks/runbenchmarks.jl --full --seed=1 --format=csv --output=before.csv
git checkout feature-branch
julia -O3 -t 16 benchmarks/runbenchmarks.jl --full --seed=1 --format=csv --output=after.csv
```

Use `--no-io` to skip filesystem-dependent cases and `--no-hdc` to skip the TEXT HDC workloads.

## GitHub Actions

The repository includes two workflows:

- `.github/workflows/benchmark.yml` runs the benchmark suite on pushes, pull requests, and manual dispatches. It uploads a `benchmark-report-<sha>-<run-id>` artifact containing raw JSON, metadata, and a Markdown summary.
- `.github/workflows/benchmark-pages.yml` downloads retained benchmark artifacts from successful `main` branch benchmark runs, aggregates them, and deploys a GitHub Pages dashboard.

Manual runs can choose a heavier mode:

```shell
gh workflow run benchmark.yml -f mode=full
gh workflow run benchmark.yml -f mode=exhaustive -f repeats=9 -f min_sample_ms=50
```

GitHub Actions artifacts are retained according to the repository retention policy. The workflow requests 90 days, so the Pages dashboard shows all retained benchmark artifacts available to the Pages build.

## Local Dashboard Build

After running or downloading benchmark artifacts locally, build the static dashboard with:

```shell
node benchmarks/site/build.js site --artifacts .benchmark-artifacts --site benchmarks/site --out public
```

Serve `public/` to inspect the same table that GitHub Pages publishes:

```shell
python -m http.server 8000 --directory public
```

Then open `http://localhost:8000`.
