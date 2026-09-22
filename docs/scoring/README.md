# Python checkpoint datasets

`deltx dataset /path/to/repository --output dataset.csv` builds one row per
meaningful Python checkpoint. `--no-filter` and `-no-filter` process every
reachable checkpoint. Both modes analyze Python only, including Python tests.

The existing `deltx-extract` command still produces the original AI-only
Parquet dataset. The `deltx extract` alias invokes that same implementation.
`python -m deltx.extraction.cli dataset ...` works before reinstalling entry
points. Dataset extraction owns the complete scan, context and scoring workflow.

## Local Docker workflow

```bash
docker compose -f docker/sonarqube/compose.yaml up -d sonarqube
# Open http://localhost:9000, complete initial setup, and create a user token
# with Browse, Execute Analysis, and project-creation permission if needed.
# Set SONAR_TOKEN in the environment or your untracked .env file.
poetry run deltx dataset /path/to/python-repository --output dataset.csv
poetry run deltx dataset /path/to/python-repository --output all.csv --no-filter
```

The pinned research stack is SonarQube **9.9.8-community** and SonarScanner
**5.0.1** (Java 17 in the scanner image). This deliberately retains the
INFO/MINOR/MAJOR/CRITICAL/BLOCKER severity contract. It is a historical research
baseline, not a recommendation to expose this server publicly. Compose binds
port 9000 to loopback and keeps data, extensions, logs and a persistent Docker
network. The single-node embedded database is for local research.

The manager probes `SONAR_HOST_URL` (default `http://localhost:9000`). A healthy
existing server is preserved. An existing starting server is awaited. If the
default endpoint refuses connections, Deltx starts its Compose stack and waits
for readiness. Timeouts, authentication errors and malformed responses fail
without starting another server. It never stops servers, including servers it
started. Docker must be available even when the server already runs because
the scanner is Dockerized.

For a newly started stack, the scanner uses Docker DNS (`sonarqube:9000`). For
an existing local server it uses host networking on Linux (including loopback-only
servers) and `host.docker.internal` with the host-gateway mapping on Docker Desktop.
`SONAR_SCANNER_HOST_URL` can override container routing; for example,
use `http://host.docker.internal:9000` when a custom container setup needs it.
`SONAR_COMPOSE_FILE` locates the Compose file for installations outside this
checkout. `SONAR_EXPECTED_VERSION`, `SONAR_SCANNER_IMAGE`, request/startup/scan/
Compute Engine timeouts, and polling interval are typed settings in
`scoring/sonarqube/config.py`. Version changes require deliberate configuration
and compatible severity metadata; unfamiliar severities are rejected.

Authentication is a secret setting. The scanner receives `sonar.login` in a
temporary mode-0600 settings file, not in the host command line. It runs as the
host UID/GID on Unix. The settings file and scan work directory are removed
afterward, and subprocess diagnostics redact the token. No Java or scanner
installation is needed on the host.

Each scan sees a fresh snapshot of tracked, regular `.py` files only. Symlinks,
submodules and generated/environment directories are omitted; tests remain.
Repository scanner settings and scripts are not loaded or executed. The scanner
sets `sonar.sources=.`, `sonar.inclusions=**/*.py`, the commit SHA as project
version and SCM revision, and a stable project key derived from the canonical
local repository path. SCM discovery is disabled for these blob snapshots.

After upload Deltx reads the submitted `ceTaskId`, waits for SUCCESS, and rejects
FAILED, CANCELED, timeout or malformed responses. It verifies the latest analysis
ID before and after collection and rejects server-version or Python-profile
changes across checkpoints. API redirects are rejected. Do not run concurrent
jobs against the same project key. HTTP requests are bounded; active issues use
`resolved=false` and pagination. Above Sonar's 10,000-result cap, queries partition by file; an
unretrievable partition or inconsistent total fails rather than truncating data.

The API retrieves `ncloc`, `cognitive_complexity`, `duplicated_lines_density`,
`sqale_index` and `sqale_debt_ratio`. Missing required measures on nonempty code
fail. Missing debt is recorded as absent and contributes no debt synthetic mark.
No Sonar A–E rating is used as a Deltx target.

## Separation and temporal policy

- `common`: exceptions, safe subprocess boundary, namespaced settings, canonical
  15-feature model and shared Python scope.
- `extraction`: Git checkpoints, semantic fingerprints, churn, import graph,
  detector invocation, orchestration and CSV serialization.
- `scoring/sonarqube`: HTTP, Docker lifecycle, scanner and completed analyses.
- `scoring/squale`: pure formulas, explicit mappings and four-score aggregation.
- `scoring/config.py`: versioned, typed research parameters with immutable
  nested mappings and stable JSON serialization.

A temporary local clone shares Git objects but never writes the user's `.git`,
index, HEAD, worktree registry or working files. Commit blobs are materialized
into temporary snapshots; no destructive checkout occurs. Traversal captures
HEAD once, then uses deterministic reverse topological order (ancestors first).
All reachable commits, including merged branches, are considered. Each diff
uses the first parent; roots use Git's empty-tree semantics. Commit timestamps
are author timestamps for provenance, not a guarantee of monotonic wall time
when Git history has skewed clocks.

Shallow repositories are rejected because their missing ancestors would turn
partial history into an apparent root and understate historical churn.
First-parent SHA and traversal policy are recorded alongside each row. Adjacent
rows can belong to sibling branches; their position alone is not a mainline
time series. Build forecasting sequences along a chosen ancestry path and use
ordered splits that keep related commits and overlapping windows together.

AST fingerprints remove recognized module/class/function/async-function
docstrings and ignore location attributes. Comments, formatting, docstring-only
changes and equivalent renames are skipped. Added/deleted nonempty normalized
ASTs and one-line semantic changes are retained. Historical parse failures fall
back to tokens retaining NEWLINE/INDENT/DEDENT; failed tokenization preserves the
raw source, conservatively retaining uncertain changes. Other runtime uses of
docstrings are outside this intentionally specified semantic policy.

Churn uses Git numstat for retained Python changes (all changed Python files
with no-filter). Non-Python content never contributes. A rename is one logical
file change. Deletions retain their deleted LOC. Crossing a `.py` boundary counts
the full Python side as added/deleted. Binary Python blobs fail explicitly.
Python encoding declarations are honored before the legacy UTF-8/Latin-1
fallback. Tracked generated Python files still count toward the explicitly
requested all-Python Git churn, but are excluded from AI, topology and Sonar snapshots.

The graph is a **static import dependency graph**, not a runtime call graph.
Nodes are files; edges point from importer to in-repository imported modules.
Root/src layouts, relative imports and package initializers are supported.
Dynamic imports and unparseable historical sources have no inferred outgoing
edges. Raw PageRank is averaged over meaningfully changed files; removed or
excluded files contribute zero. If none changed meaningfully, the mean is zero
even under no-filter. Ties use upper ECDF ranks; a singleton has percentile 1.

AI inference uses `AIDetectionInference.analyze_commit` on surviving changed
Python sources, with the actual commit identity/timestamp. Its result already
contains a LOC-weighted **percentage**, and is not rescaled. Existing detector
skip rules for setup.py/conftest.py remain intact; those files still participate
in Sonar analysis and topology. No scoreable AI evidence produces the detector's
existing `0.0` fallback, flagged in metadata. It is not proof of human authorship.

## Exact scoring equations

For rule count `count(r,t)` and current `ncloc`,

```text
density(r,t) = count(r,t) / max(ncloc, 1) * 1000
F'(r,t) = ln(1 + density) / (1 + ln(1 + density))
C'(f,t) = ECDF(PR(f,t))
V(f,t) = sum(k=1..H) delta^(k-1) *
         (added(f,t-k) + deleted(f,t-k)) / max(previous_loc(f,t-k), 1)
CH'(f,t) = ln(1 + V) / (1 + ln(1 + V))
W(i,d,t) = M(i,d) * S(i) *
           [1 + rho_d * (alpha_d*F' + beta_d*C' + gamma_d*CH')]
IM(i,d) = 3 * [1 - clip(W / (5*(1+rho_d)), 0, 1)^kappa_d]
metric_mark(x) = 3 / [1 + (x/tau)^k]
M_d = -ln(sum(omega_j * lambda_d^(-IM_j)) / sum(omega_j)) / ln(lambda_d)
score_d = clip((100/3) * M_d, 0, 100)
```

Historical volatility uses physical file LOC and all prior first-parent
checkpoints, including filtered checkpoints. Renames preserve file identity;
newly added files do not inherit unrelated deleted-file history. The current
change and future/sibling commits cannot enter `V(f,t)`.

The severity baseline is INFO=1, MINOR=2, MAJOR=3, CRITICAL=4, BLOCKER=5.
Overrides map rule keys to one or more dimensions with coefficients in `(0,1]`;
explicit overrides win. Fallbacks are BUG→CORRECTNESS,
VULNERABILITY/SECURITY_HOTSPOT→SECURITY, CODE_SMELL→MAINTAINABILITY.
Unknown types without overrides are listed in metadata and logged.

The initial curated efficiency override is `python:S2190` (unbounded recursion):
it influences correctness and efficiency with M=1 each, accounting for wasted
CPU/stack resources as well as failure. This is an explicit Deltx research
mapping, not a Sonar efficiency classification. Broader performance/resource
coverage needs review and calibration. Message keywords never determine mapping.
The selected issues endpoint does not expose security hotspots with this severity
contract; separate hotspot review-priority values are **not** converted into
invented severities, so those hotspots are outside this baseline's scores.

Maintainability adds marks for `sqale_index/ncloc*1000` (debt minutes/KLOC),
`cognitive_complexity/ncloc*1000`, and duplication percentage, even with no smells.
The one-line denominator applies when ncloc is zero. No issue/synthetic marks
means score 100. Empty Python states are still scanned; absent empty-state metrics
become zero, AI/churn/PageRank follow their documented rules, and empty-state
quality scores are 100. These mean no measured findings, not assessed excellence.

The weighted exponential mean penalizes low marks more than arithmetic averaging.
It is not a hard worst-issue cap: an arbitrarily large population of better marks
can dilute a bad mark under the prescribed normalized-mean equation. Log-sum-exp
and expm1/log1p keep extreme weights/bases numerically stable. Nonfinite model
values are rejected before writing CSV.

## Baseline configuration and output

`PYTHON_RESEARCH_BASELINE_V1` uses equal alpha/beta/gamma=1/3, rho=1, kappa=1,
lambda=9 and omega=1 per dimension. Synthetic tau values are 1000 debt minutes/KLOC,
100 cognitive complexity/KLOC and 5% duplication, with k=1 and omega=1. H=50,
delta=0.9, PageRank damping=0.85, tolerance=1e-12 and max iterations=1000.
These are provisional baselines, **not Python-optimized or validated thresholds**.

Export/edit a complete config and use `--scoring-config baseline.json`:

```python
from pathlib import Path
from deltx.scoring.config import ScoringConfig

Path("baseline.json").write_text(ScoringConfig().model_dump_json(indent=2))
```

The model CSV has exactly these columns, in order, with no index or identity fields:

```text
score_maintainability,score_correctness,score_security,score_efficiency,
ai_confidence_pct,loc_added,loc_deleted,files_modified_count,
avg_pagerank_centrality,density_blocker_issues,density_critical_issues,
density_major_issues,density_minor_issues,cognitive_complexity,duplication_density
```

The separate `<name>.metadata.csv` records matching row index, commit SHA/time,
first parent, traversal policy, repository and captured HEAD, filtering,
server/profile/scanner and Python versions, analysis ID, config version/full
JSON/hash, AI evidence and unknown rules. Keep these fields outside the feature
matrix. Both files are staged before publication; analysis failure preserves
existing outputs. An ordinary publication error restores the previous pair;
if restoration also fails, the error identifies retained recovery copies.
Two file replacements are not crash-atomic: consume outputs after successful
completion and use a separate destination per concurrent job. No resume is
implemented for this full dataset command. Existing AI-only Parquet resume is
unchanged.

The scores are derived from same-checkpoint issue densities, complexity and
duplication. Predicting those scores from the same row is circular; forecasting
must align input history with later targets. Fit thresholds, mappings, scalers
and any other learned preprocessing on training history only. Exclude or flag
rows without AI evidence instead of interpreting their fallback zero as a
human-authored label.

Run `poetry run pytest --cov-fail-under=80`, `poetry run ruff check .`,
`poetry run ruff format --check .`, and `poetry run mypy src tests`. Unit tests
use real temporary Git histories and mocked Docker/HTTP/detection. An optional
real scanner test runs only with `DELTX_RUN_SONAR_INTEGRATION=1` and a configured
token; it does not download detector weights.

External issue triage can persist across scans, including prior severity edits
and accepted findings. Use a dedicated project with no manual issue edits and
keep its analysis settings fixed; the API checks do not prove freedom from
pre-existing triage. Profile/version checks catch drift during a run, but
Python parser versions, analyzer plugins and detector versions also affect
results. Keep the recorded configuration and tool environment fixed for
comparable experiments. Pinned older Sonar analyzers may not understand newer
Python syntax; the semantic filter's conservative fallback does not upgrade
Sonar's parser.

References: [SonarScanner 9.9 documentation](https://docs.sonarsource.com/sonarqube-server/9.9/analyzing-source-code/scanners/sonarscanner),
[SonarSource Python S2190 announcement](https://community.sonarsource.com/t/python-analysis-detects-more-tricky-quality-issues-unused-assigned-variables-infinite-loops-ignored-parameters-initial-value-and-more/17146),
[pinned server image](https://hub.docker.com/layers/library/sonarqube/9.9.8-community/images/sha256-f5f29a61164204ea3ffd8c9ad74413e2c06c94823f2c671b548cb0b916caba4f),
[pinned scanner Java 17 image](https://hub.docker.com/layers/sonarsource/sonar-scanner-cli/5.0.1/images/sha256-02372948eaeeb10dfbe0cfd4174d44b8e405d0aeae431532b2bdb21d0347bf23).
