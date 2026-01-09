# Repository Guidelines

## Project Structure & Module Organization
- `agentless/`: core pipeline. `fl/` handles localization, `repair/` generates/reranks patches, `test/` runs regression/reproduction harnesses, and `util/` holds shared helpers.
- `classification/`: SWE-bench Lite classification utilities and CSV outputs.
- `get_repo_structure/`: preprocessing scripts for benchmark repo structures.
- `data/`: local datasets (e.g., `Loc-Bench_V1_dataset.jsonl`).
- `locbench_repos/`: local mirrors of target repos (directory names use `org_repo`, e.g., `django/django` → `django_django`).
- `evaluation/`: localization outputs and scoring utilities (see `evaluation/loc_output/` and `evaluation/eval_metric.py`).
- `dev/`: small utilities such as cost reporting.
- `resources/`: static assets used by documentation.
- `results/` and `logs/`: generated artifacts (gitignored); keep large JSON/JSONL outputs out of commits.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`
- Enable local imports: `export PYTHONPATH=$PYTHONPATH:$(pwd)`
- Formatting hooks: `pre-commit install`
- Example localization run:
  `python agentless/fl/localize.py --file_level --output_folder results/swe-bench-lite/file_level --num_threads 10`
- Example repair run:
  `python agentless/repair/repair.py --loc_file <locs.jsonl> --output_folder results/swe-bench-lite/repair_sample_1`
- Utility example: `python dev/util/cost.py --output_file results/.../output.jsonl`

## Datasets & Localization Outputs
- `data/Loc-Bench_V1_dataset.jsonl` entries include `repo`, `instance_id`, `base_commit`, `problem_statement`, `patch`, `test_patch`, `edit_functions`, `labels`, and `category`.
- Local repo copies live under `locbench_repos/` and should match the `repo` field with `/` replaced by `_`.
- Loc-Bench localization outputs are stored under `evaluation/loc_output/` (example: `evaluation/loc_output/locagent/claude_3-5/loc_outputs.jsonl`).
- Evaluation expects prediction fields like `found_files`, `found_modules`, and `found_entities`, plus `raw_output_loc` and `meta_data` for traceability.

## Coding Style & Naming Conventions
- Python, 4-space indentation; keep modules and functions in `snake_case`, constants in `UPPER_CASE`.
- Formatting is enforced via pre-commit: `black` and `isort` (profile `black`). Run `pre-commit run --all-files` before opening a PR.
- Output artifacts typically use JSONL naming like `output_<n>_processed.jsonl`.

## Testing Guidelines
- Testing is benchmark-driven and uses the SWE-bench harness (Docker required).
- Regression and reproduction runners live in `agentless/test/` (see `README_swebench.md` for full pipelines).
- Keep test outputs under `results/` and avoid committing generated artifacts.

## Commit & Pull Request Guidelines
- Commit history uses conventional prefixes like `feat:`, `fix:`, and `doc:`; keep the subject line short and imperative.
- PRs should include a brief summary, benchmark/dataset targeted (e.g., SWE-bench Lite), commands run, and key output paths or logs.

## Configuration & Secrets
- Export provider keys via environment variables (e.g., `OPENAI_API_KEY`); never commit secrets.
- Optional data cache paths can be set via env vars like `PROJECT_FILE_LOC` (see `README_swebench.md`).
- `agentless/fl/localize.py` loads `.env` by default; set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `LOCBENCH_DATASET_PATH`, and `LOCBENCH_REPO_ROOT` there for local runs.
