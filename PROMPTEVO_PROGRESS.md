# PromptEvo — Implementation Progress

Status: **core implementation complete; stopped because the work finished, not because of context limits.**

## 1. Project Overview
PromptEvo evolves a six-slot structured prompt against a frozen worker LLM under a global worker-call budget, logging per-child, per-operator effects. See README.md.

## 2. Current Implementation Status
| Component | Status | Details |
|---|---|---|
| Project structure | ✅ | Spec layout + `llm/`, `viz.py`; pyproject.toml |
| Structured prompts | ✅ | parse (heuristic + JSON/YAML), serialize, hash |
| Instruction wording operator | ✅ | `operators/instruction_wording.py` |
| Instruction ordering operator | ✅ | permutation-validated |
| Constraints operator | ✅ | |
| Few-shot operator | ✅ | ids resolved from dev/pool |
| Output format operator | ✅ | must contain "answer" |
| Reasoning operator | ✅ | |
| Crossover | ✅ | LLM, provenance recorded |
| Feedback operator | ✅ | optional, off by default |
| Evaluation harness | ✅ | classification/math/qa metrics, caching |
| Budget manager | ✅ | worker cap, optimizer tracked, cost |
| Racing | ✅ | bound + Wilson rules |
| Bootstrap evaluation | 🟡 | resamples cached outcomes (no extra calls); not re-querying worker |
| Selection | ✅ | tournament, Pareto, robust score |
| Operator logging | ✅ | per-child log + stats + CIs |
| Baselines | ✅ | seed, manual (unavailable if absent), genetic |
| Experiments | ✅ | A, B1, B2, C, transfer, Thompson |
| CLI | ✅ | optimize/evaluate/baseline/experiment/analyze |
| Tests | ✅ | 49 passing |
| README | ✅ | |

## 3. Files Created
(All new; repository was empty.) `prompt_evo/`: `config.py` (pydantic YAML config), `utils.py`, `cli.py`, `__main__.py`, `viz.py` (8 plot types);
`schemas/{prompt,dataset,evaluation,evolution}.py`; `llm/{base,openai_compat,factory}.py`;
`prompts/{parser,serializer,mutation_templates,crossover_templates}.py`;
`operators/{base,instruction_wording,instruction_ordering,constraints,few_shot,output_format,reasoning,crossover,feedback}.py`;
`evaluation/{evaluator,metrics,racing,bootstrap,slices}.py`;
`core/{optimizer,population,individual,selection,budget,operator_selector}.py`;
`logging/{operator_logger,evolution_logger}.py`; `baselines/{seed,manual,genetic}.py`;
`experiments/{runner,ablation,robustness,significance}.py`.
Other: `tests/` (conftest, fakes.py, toy_tasks.py, 5 test files), `configs/{example,experiment}.yaml`, `data/*.jsonl` (60 dev + 60 test per task), `prompts/*.txt`, `scripts/make_toy_data.py`, README.md, pyproject.toml.
Not created as separate files (merged): spec's `core/..` OK; `evaluation/slices.py` holds slicing; adaptive selection is in `core/operator_selector.py`.

## 4. Files Modified
None pre-existing.

## 5. Architecture
Config → `build_task` → `PromptEvo` (Evaluator + BudgetManager + Selection + operators + OperatorLogger) → `RunResult` → `save_run`. Experiments call `run_method` per (method, seed). See README mermaid diagram.

## 6. Algorithms Implemented
Steady-state (mu+1) evolution; init by round-robin mutation of the seed (+crossover when ≥2 members); tournament/Pareto parents; mutation/crossover/feedback chosen by normalized probabilities; fixed or Thompson operator choice; nested staged evaluation with promotion + racing; global worker budget; length penalty; bootstrap CI/std; duplicate detection via normalized-prompt hash (includes discarded prompts); generations = `population_size` children.

## 7. Six Mutation Operators
All six: fully implemented, LLM template in `prompts/mutation_templates.py`, JSON `{new_component, reason}`, pydantic/slot validation, retry-with-feedback then discard (logged `invalid_mutation`). Tested against a deterministic fake optimizer only.

## 8. Evaluation System
`WorkerModel.generate(prompt, input)`; `Evaluator` (cache, racing, timeout, budget reservation before each call); metrics `classification|math|qa`; datasets JSON/JSONL/CSV; per-example `ExampleResult` incl. error category; worker failures raise `EvaluationError` (run aborts after 5 consecutive).

## 9. Operator Logging
Schema: `OperatorLogEntry` (see README). Answer "which mutation types improve performance": `prompt_evo analyze --results <dir>` or `operator_stats.json` (mean/median delta, win rate, CI, fixed/broken per operator).

## 10. CLI
`python -m prompt_evo.cli optimize --config configs/example.yaml` · `evaluate --prompt P --dataset D` · `baseline --method genetic --config C` · `experiment --config C --kind A|B1|B2|C|transfer|all` · `analyze --results DIR [--plots] [--full-only]`.

## 11. Configuration
Sections `task, optimizer, evaluation (+racing), worker, optimizer_llm, experiment`; strict (unknown keys rejected). See `prompt_evo/config.py`, `configs/*.yaml`.

## 12. Tests
49 tests, all passing (`pytest -q`, ~3 s): prompt parsing/serialization/duplicates, each operator, crossover, invalid/empty/garbage LLM output + retry budgets, metrics, evaluator/cache, budget & cost accounting, worker failure, racing, bootstrap, slices, selection (tournament/robust/Pareto), Thompson/fixed selectors, dataset loading/validation, operator-stat math, e2e run, log invariants, same-seed reproducibility, tiny/exhausted budgets, single-operator isolation, feedback+adaptive, length penalty, baselines, Experiment A artifacts, B1/B2.

## 13. Experiments
Evolution vs manual: IMPLEMENTED · Operator isolation (B1): IMPLEMENTED · Leave-one-out (B2): IMPLEMENTED · Robustness (tasks×workers×budgets): IMPLEMENTED · Budget comparison: IMPLEMENTED (inside C) · Adaptive selection: IMPLEMENTED (optional flag; no dedicated comparison experiment) · Transfer: IMPLEMENTED.

## 14. Current Limitations
- **Never run against a real LLM**; OpenAI client untested (no network/keys).
- Toy results use a *simulated* worker/optimizer (`tests/fakes.py`): they validate plumbing only, not scientific claims.
- Seeds=3 gives wide CIs (B2 intervals include 0). The two toy configs gave different PromptEvo means (0.644 vs 0.772 test acc), showing high variance.
- Few-shot pool defaults to dev set (leakage). Bootstrap resamples existing outcomes. `all_evaluated` operator stats are biased by partial-slice children. Crossover often yields duplicates (success_rate low). No lint/type-check run. Not committed to git (repo initialised, nothing committed).

## 15. Exact Next Steps
1. Run with a real worker/optimizer on a real dataset (50–200 dev examples); verify prompts/templates with real models.
2. Add a separate few-shot pool or exclude few-shot ids from dev scoring.
3. Run ≥5–10 seeds; report B2/C with CIs.
4. Add ruff/mypy and fix findings.
5. Dedicated Thompson-vs-fixed experiment.
6. Parallel evaluation for API throughput.

## 16. How to Continue
Start: "Read PROMPTEVO_PROGRESS.md and README.md; run `pytest -q`; then connect a real LLM via configs (provider: openai)". Nothing is mid-edit. Decisions to keep: only fully dev-evaluated children enter the population; paired parent-vs-child deltas; fixed operator probabilities by default; test set never used for selection; `tests/fakes.py` is the only simulated LLM code; seed evaluated in normalized serialized form.

## 17. Validation
- `pytest -q` → 49 passed.
- `python -m prompt_evo.cli optimize --config configs/example.yaml` → seed dev 0.55 → ran to 1000 worker / 26 optimizer calls.
- `experiment --config configs/example.yaml --kind A` (3 seeds, simulated worker; test acc): seed 0.550, manual 0.633, genetic 0.639±0.025, PromptEvo 0.644±0.010 (example.yaml: population 8, 20 generations). The same command with configs/experiment.yaml (population 6, 10 generations) gave PromptEvo 0.772±0.051 — different configs, same seeds, so variance across configs is large..
- `experiment --config configs/experiment.yaml --kind all` completed: A: seed 0.550, manual 0.633, genetic 0.639, PromptEvo 0.772 (paired permutation p<0.001 vs seed/manual/genetic, d_z≈0.46–0.58, n=60 test examples, simulated worker); B1 and B2 ran (B2 CIs include 0); C: best operator differed across tasks/workers/budgets; transfer ran.
- Lint/type-check: not run.
