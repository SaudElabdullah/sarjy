.PHONY: install check lint type test test-integration run db-start db-reset db-push evals \
	evals-offline evals-dry evals-staging deploy-staging deploy-prod smoke-staging smoke-prod \
	static-staging static-prod release-staging release-prod

install:
	uv sync --all-extras

lint:
	uv run ruff check . && uv run ruff format --check .

type:
	uv run mypy

test:
	uv run pytest -q

test-integration:
	uv run pytest -q -m integration

check: lint type test

run:
	uv run uvicorn sarjy.main:create_app --factory --reload --port 8000

db-start:
	supabase start

db-reset:
	supabase db reset

db-push:
	supabase db push

# Full suite: real Gemini, mock weather. Gates at red-team 99% / benign FP 2%.
evals:
	uv run python tests/evals/run_evals.py

# Layer-2-only run of the red-team and benign suites. No API key, no database:
# an LLM that always complies, so every refusal measured came from a guard.
evals-offline:
	uv run python tests/evals/run_evals.py --offline

# Shape + coverage-matrix check of every eval JSONL. No network.
evals-dry:
	uv run python tests/evals/run_evals.py --dry-run

# Phase 8 Task 5: staging/prod deploy, smoke test, static-client publish.
# Manual deploy path (no CI): run these in order from a shell that has the
# Fly/Supabase vars exported.
# FLY_API_TOKEN must be set (flyctl auth or `fly auth token`); smoke-* need
# SUPABASE_URL/SUPABASE_ANON_KEY; static-* need SUPABASE_URL/
# SUPABASE_SERVICE_ROLE_KEY (or the STAGING_/PROD_-prefixed variants — see
# scripts/upload_static.py).
deploy-staging:
	flyctl deploy -c fly.staging.toml --remote-only

deploy-prod:
	flyctl deploy -c fly.prod.toml --remote-only --strategy rolling

smoke-staging:
	uv run python scripts/smoke.py https://sarjy-staging.fly.dev "$$SUPABASE_URL" "$$SUPABASE_ANON_KEY"

smoke-prod:
	uv run python scripts/smoke.py https://sarjy-prod.fly.dev "$$SUPABASE_URL" "$$SUPABASE_ANON_KEY"

static-staging:
	uv run python scripts/upload_static.py --env staging

static-prod:
	uv run python scripts/upload_static.py --env prod

# Live eval suite against a deployed target, gating the manual release flow's
# staging -> prod step. SUPABASE_URL/SUPABASE_ANON_KEY/GEMINI_API_KEY (and
# TURNSTILE_TOKEN if the target has captcha on) must be in the shell env.
evals-staging:
	uv run python tests/evals/run_evals.py --target https://sarjy-staging.fly.dev --suites redteam,benign,memory,weather,ocean

# Manual release flow (no CI): run `supabase link` once first, then this
# does db push -> deploy -> static publish -> smoke test for one environment.
release-staging: db-push deploy-staging static-staging smoke-staging

release-prod: db-push deploy-prod static-prod smoke-prod
