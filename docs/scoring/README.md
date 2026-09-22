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
# For a new checkout, copy .env.example to .env. Keep an existing .env.
# Set SONAR_HOST_URL in .env, then:
poetry run deltx sonar up
# Open SONAR_HOST_URL, complete initial setup, and create a User token
# with Browse, Execute Analysis, and project-creation permission if needed.
# Set SONAR_TOKEN in the environment or your untracked .env file.
poetry run deltx dataset /path/to/python-repository --output dataset.csv
poetry run deltx dataset /path/to/python-repository --output all.csv --no-filter
poetry run deltx sonar down  # retains volumes
```

The stack uses **`sonarqube:latest`** (Community Build) and
**`sonarsource/sonar-scanner-cli:latest`**. Compose pulls the server when starting
the stack. The scanner image is pulled once per analyzer instance and then
addressed by its immutable image ID for every checkpoint. Actual server/scanner
versions, scanner image ID and Python quality profile are recorded in the sidecar.
The single-node embedded database is for local research.

`SONAR_HOST_URL` is required in `.env` (or an environment override). There is no
separate scanner URL or port setting. `deltx sonar up/down` derives the loopback
binding and web context from this URL and passes them to Compose. Use these
commands instead of invoking Compose without the derived environment. HTTP
servers can be started automatically; HTTPS servers must be provided separately.

The manager probes `SONAR_HOST_URL`. A healthy existing server is preserved.
An existing starting server is awaited. If the configured endpoint refuses
connections, Deltx starts its Compose stack and waits
for readiness. Timeouts, authentication errors and malformed responses fail
without starting another server. Dataset runs do not stop servers. The explicit
`sonar down` command stops only the managed stack and keeps its volumes.
Docker must be available even when the server already runs because
the scanner is Dockerized.

The scanner uses the same URL with host networking on Linux (including
loopback-only servers). Docker Desktop substitutes `host.docker.internal` for
the loopback hostname while preserving the configured scheme, port and path.
`SONAR_COMPOSE_FILE` locates the Compose file for installations outside this
checkout. An optional `SONAR_EXPECTED_VERSION` constrains the server version for
repeat experiments. `SONAR_SCANNER_IMAGE` can select a registry image digest.
Request/startup/scan/Compute Engine timeouts and polling interval are typed settings
in `scoring/sonarqube/config.py`. Community Build 25+ and Server 2025+ version
formats are accepted; the live integration test targets the current `latest`
image. Unfamiliar severities or software qualities fail explicitly.

Authentication is a secret setting. The HTTP client uses Bearer authentication;
the public readiness endpoint is queried without credentials.
The scanner receives `sonar.token` in a temporary mode-0600 settings file,
not in the host command line. It runs as the
host UID/GID on Unix. The settings file and scan work directory are removed
afterward, and subprocess diagnostics redact the token. No Java or scanner
installation is needed on the host.

### Moving from the old 9.9 stack

The Compose project is now `deltx-sonarqube-current`, with new data, extension
and log volumes. The old `deltx-sonarqube_sonar-*` volumes are retained. This
starts a fresh server with the current built-in Python profile and requires a
new User token; a token from the old server will not authenticate. Stop an old
server occupying `SONAR_HOST_URL` before starting the new one. Existing datasets
are unchanged; regenerate them into a new output to compare the new baseline.

If the old Deltx stack is still running, stop it without deleting its volumes:

```bash
COMPOSE_PROJECT_NAME=deltx-sonarqube poetry run deltx sonar down
poetry run deltx sonar up
```

To retain old server accounts, profiles and analysis history, follow SonarSource's
[supported database update path](https://docs.sonarsource.com/sonarqube-community-build/server-update-and-maintenance/update/determine-path)
with backups and intermediate versions. Do not mount the 9.9 data or old plugin
volume directly into `latest`. Subsequent server updates also need the supported
database path; `latest` is a moving tag, not a reproducibility guarantee.

### Analysis and collection

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
`components=<project>` and `issueStatuses=OPEN,CONFIRMED` with pagination.
Accepted, false-positive, fixed and sandbox findings are excluded.
Above Sonar's 10,000-result cap, queries partition by file; an
unretrievable partition or inconsistent total fails rather than truncating data.

The API retrieves `ncloc`, `cognitive_complexity`, `duplicated_lines_density`,
`software_quality_maintainability_remediation_effort` and
`software_quality_maintainability_debt_ratio`. These populate the domain's
`sqale_index` and `sqale_debt_ratio` fields; legacy measures are a fallback if
MQR values are absent. NCLOC, debt, cognitive complexity and duplication are
required, including for empty snapshots. Explicit measured zero is valid;
missing values fail instead of becoming zero. Only the unused debt ratio is optional.
No Sonar A–E rating is used as a Deltx target.

The client loads the active Python rules through paginated `api/rules/search`
queries for the project's quality profile. It parses rule attributes into an
immutable `RuleCatalog`, cached by profile identity and update timestamp for the
analyzer's lifetime. No per-issue metadata requests occur. Profile identity is
checked again after collection to reject changes while loading rules. Missing
rule metadata, inconsistent counts, unfamiliar attributes and zero active
`EFFICIENT` rules fail clearly. The active rule and efficiency-rule counts are logged.

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
W(i,d,t) = M(i,d) * S(i,d) *
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

The V3 baseline prefers MQR impacts even when legacy fields are also returned.
MAINTAINABILITY→MAINTAINABILITY, RELIABILITY→CORRECTNESS and SECURITY→SECURITY.
Each quality keeps its own impact severity; an issue can affect multiple scores.
Repeated impacts for one dimension reduce to their strongest severity, independent
of ordering, so one issue contributes at most once to each dimension.
MQR INFO/LOW/MEDIUM/HIGH/BLOCKER normalize to the existing
INFO/MINOR/MAJOR/CRITICAL/BLOCKER buckets (1/2/3/4/5). The four CSV density
columns keep their names and count each issue once at its maximum impact.
Unknown severities fail rather than receiving a default weight. Do not mix rows
from different scoring baselines as if they used the same targets.

`rule_overrides` can configure coefficients in `(0,1]` for dimensions already
mapped from issue impacts and rule metadata. They cannot add dimensions or
replace any modern impact. Defaults contain no rule-ID overrides. If impacts
are absent, compatibility fallbacks are BUG→CORRECTNESS,
VULNERABILITY/SECURITY_HOTSPOT→SECURITY, CODE_SMELL→MAINTAINABILITY.
Unknown types without impacts or an efficiency classification are listed in
metadata and logged.

Efficiency is an **additive** mapping, exclusively for rules whose authoritative
`cleanCodeAttribute` is `EFFICIENT`. It uses the strongest severity among the
issue's modern impacts, or its valid legacy severity when modern impacts are
absent. It retains all other software-quality impacts. Rule IDs, keywords,
tags, debt, complexity and duplication cannot identify efficiency findings.
For example, `python:S2190` is `LOGICAL` in the current Python analyzer and thus
does not receive an efficiency penalty. An active profile with efficient rules
and no corresponding issue violations yields efficiency 100; a profile with
no active efficient rules raises a configuration error.

Efficiency is a **static-analysis-derived efficiency proxy**, not a runtime
performance measurement. Correctness and security similarly score 100 when
their configured rules report no violations.
Only findings returned by the issues API enter scoring. Security findings with
MQR impacts contribute normally. Separate hotspot review priorities are not
converted into issue severities.

Maintainability first combines all MAINTAINABILITY issue marks using nonlinear
SQUALE into one **0–3 issue-practice mark**, or 3 when no such issues exist.
The final SQUALE aggregation combines exactly four practice marks:

1. The issue-practice mark, weighted by `maintainability_issue_omega`.
2. `metric_mark(sqale_index / max(ncloc, 1) * 1000)` for debt minutes/KLOC.
3. `metric_mark(cognitive_complexity / max(ncloc, 1) * 1000)`.
4. `metric_mark(duplicated_lines_density)` for the already normalized percentage.

Each metric uses its own configured positive `omega`, `tau` and `k`. Duplication
is not divided by NCLOC. Issue count affects frequency risk but cannot change
the relative weights of the four outer practices. No maintenance issues alone
does not imply a perfect score: high debt, complexity or duplication lowers it.
Empty Python states are still scanned, but cannot be scored if Sonar omits any
required measure. Missing data never earns a perfect mark.

The weighted exponential mean penalizes low marks more than arithmetic averaging.
It is not a hard worst-issue cap: an arbitrarily large population of better marks
can dilute a bad mark under the prescribed normalized-mean equation. Log-sum-exp
and expm1/log1p keep extreme weights/bases numerically stable. Nonfinite model
values are rejected before writing CSV.

## Baseline configuration and output

`PYTHON_RESEARCH_BASELINE_V3` uses equal alpha/beta/gamma=1/3, rho=1, kappa=1,
lambda=9 and omega=1 per dimension. Synthetic tau values are 1000 debt minutes/KLOC,
100 cognitive complexity/KLOC and 5% duplication, with k=1 and omega=1.
`maintainability_issue_omega=1` weights the issue practice at the outer level. H=50,
delta=0.9, PageRank damping=0.85, tolerance=1e-12 and max iterations=1000.
These are provisional baselines, **not Python-optimized or validated thresholds**.

Export/edit a complete config and use `--scoring-config baseline.json`:

```python
from pathlib import Path
from deltx.scoring.config import ScoringConfig

Path("baseline.json").write_text(ScoringConfig().model_dump_json(indent=2))
```

The dataset CSV has 17 columns: two string identifiers followed by the 15 numeric
features. There is no extra DataFrame index column:

```text
repository,commit_hash,score_maintainability,score_correctness,score_security,score_efficiency,
ai_confidence_pct,loc_added,loc_deleted,files_modified_count,
avg_pagerank_centrality,density_blocker_issues,density_critical_issues,
density_major_issues,density_minor_issues,cognitive_complexity,duplication_density
```

`repository` contains only the repository directory name, for example `Pyevolve`.
The metadata sidecar retains the resolved absolute path for provenance.
Directory names are not globally unique: disambiguate unrelated repositories
with the same name before grouping a combined training dataset. Also avoid
placing related clones/forks with shared commits in different evaluation splits.
No remote URL or credentials are copied from Git configuration.

`commit_hash` is the full Git commit object ID, matching sidecar `commit_sha`.
It identifies an exact snapshot across filtered and unfiltered exports; a row
number would change with checkpoint selection. Hashes do not express order.
Keep each repository's exported reverse-topological row order, or join its
sidecar to restore order after shuffling. Author timestamps are not a substitute
for ancestry order. Deduplicate overlapping exports by `(repository, commit_hash)`
after selecting one consistent scoring configuration and analyzer version.

For transformer preparation, group by repository first, then select the numeric
feature schema explicitly:

```python
import pandas as pd
from deltx.common.models import CommitDataVector

frame = pd.read_csv("dataset.csv", dtype={"repository": "string", "commit_hash": "string"})
feature_columns = list(CommitDataVector.model_fields)
series_by_repository = {
    repository: rows.loc[:, feature_columns].to_numpy(dtype="float32")
    for repository, rows in frame.groupby("repository", sort=False)
}
```

`CommitDataVector` remains 15-dimensional. The identifiers control grouping and
traceability; do not encode hashes or filesystem paths as numeric input channels.
Construct input/target windows within each repository, never across a repository
boundary. For forecasting within known repositories, split in temporal order
before constructing windows so training targets never come from the test period.
For evaluation on unseen repositories, hold out whole repositories. These answer
different research questions; a random split of overlapping commit windows does
not establish either result. See the
[grouped and time-series validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html).

This header applies to newly generated CSVs. Existing 15-column exports can be
joined to their matching metadata sidecar; the writer does not rewrite old files
unless that output is explicitly selected for a new run.

The separate `<name>.metadata.csv` records matching row index, commit SHA/time,
first parent, traversal policy, repository and captured HEAD, filtering,
server/profile/scanner and Python versions, scanner image ID, issue model
(`MQR_EFFICIENT_V2`), analysis ID, config version/full
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
comparable experiments. Pulling `latest` between runs can change active rules
and scores. Profile/version checks detect changes during a dataset run; they do
not make datasets from different analyzer releases directly comparable.

References: [SonarQube Web API](https://docs.sonarsource.com/sonarqube-community-build/extension-guide/web-api),
[MQR modes and severities](https://docs.sonarsource.com/sonarqube-community-build/user-guide/code-metrics/changing-modes),
[metric definitions](https://docs.sonarsource.com/sonarqube-community-build/user-guide/code-metrics/metrics-definition),
[rule attributes and impacts](https://docs.sonarsource.com/sonarqube-community-build/quality-standards-administration/managing-rules/rules),
[server image](https://hub.docker.com/_/sonarqube),
[scanner image](https://hub.docker.com/r/sonarsource/sonar-scanner-cli).
