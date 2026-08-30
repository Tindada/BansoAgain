# Banso repository instructions

## Project context

- Banso is an experimental Python 3.12 multi-step news research agent.
- Read recent commits and the relevant implementation before making architectural changes.
- Keep responsibilities clear: policies choose and parse actions, executors perform them,
  reducers update state from observations, and synthesizers produce answers from extracted
  evidence.

## Python environment

- Use `uv sync` for dependency installation and `uv run` for Python commands.
- If the default uv cache is not writable, use `UV_CACHE_DIR=.uv-cache`.
- Do not expose, print, or commit API keys, `.env` contents, or other credentials.

## Testing

- Run the most relevant tests first, then run `uv run pytest` when the change has broad impact
  or before handing off a completed implementation.
- Tests should verify observable behavior and responsibility boundaries, not isolated prompt
  wording or deleted implementation details.
- Prefer updating existing tests and removing obsolete or redundant coverage over only adding
  new tests.
- If a test run hangs, rerun the full pytest suite with elevated permissions.
- Live provider checks and GISA evaluations are not unit tests. Do not run costly online
  evaluations unless the user explicitly requests them.

## GISA evaluation

- Historical GISA runs are stored under `runs/gisa/` and may be used to assess policy, prompt,
  budget, or synthesis changes.
- Before comparing runs, verify their case set, policy, models, budgets, and relevant runtime
  configuration. Do not infer configuration only from directory names.
- Inspect both aggregate scores and per-case changes so that isolated failures are distinguished
  from broad regressions.
- Use a small smoke run before a larger evaluation when validating a new execution path. A
  60-case run is a common budget-limited comparison, not the full GISA case set.
- Score a completed run without model or retrieval calls using:

  ```bash
  uv run --group evaluation python scripts/score_gisa.py runs/gisa/<run>/results.jsonl
  ```

## Git and handoff

- Before a commit, inspect the diff, run proportionate tests, and check for redundant tests.
- Commit only when the user explicitly asks. If the user requests inspection before each commit,
  stop after verification and wait for that inspection.
