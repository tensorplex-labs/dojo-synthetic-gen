Dojo Synthetic API – CRUSH notes

- Setup: Python 3.10+; create venv, then: pip install -e '.[dev]' (dev deps: pytest, ruff, commitizen)
- Run app: python main.py --trace --env_name dev  |  or: docker compose up -d
- Lint/format: ruff format .  |  ruff check . --fix  (CI runs ruff with pyproject.toml)
- Pre-commit: make hooks  |  python3 pre-commit.pyz run -a
- Tests (all): pytest -q
- Single test file: pytest -q test_routes.py
- Single test function: pytest -q test_routes.py::test_get_question
- Keyword match: pytest -q -k get_answer
- Packaging/install: python -m pip install .

Code style
- Imports: absolute within package (e.g., commons.*), sorted by Ruff/isort (stdlib, third-party, local).
- Formatting: Ruff formatter; 88 cols, 4-space indent, double quotes, trailing commas respected.
- Types: add explicit hints; use Pydantic v2 models for FastAPI I/O; prefer "|" unions; avoid Any where possible.
- Errors: raise specific exceptions; return structured models with success flags; avoid bare except; include messages.
- Logging: use loguru logger; no print in API/business code; never log secrets/keys.
- Naming: modules/files snake_case; functions/vars snake_case; classes PascalCase; constants UPPER_SNAKE.
- Structure: endpoints in commons/routes/*.py; logic in commons/*; avoid circular imports; keep functions small.
- Env/config: use pydantic-settings; load from .env (see .env.example); do not hardcode secrets.
- Data: use commons.cache.RedisCache; prefer async interfaces where available.
- Commits: Conventional Commits via commitizen; use cz commit (feat|fix|chore|refactor|docs...).
- JS (rare): npx eslint . (config: eslint.config.mjs).
- Docs: FastAPI docs at http://localhost:5003/docs#/.
- CI: Linter on PRs (ruff); Docker build on branches; semantic-release on main.
- Cursor/Copilot rules: none detected.
- Quick all-in-one: ruff format . && ruff check . --fix && pytest -q
