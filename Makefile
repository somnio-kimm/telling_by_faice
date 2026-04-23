.PHONY: install lint format test pre-commit build clean server

install:
	cd server && uv sync --all-extras --dev
	cd server && uv run pre-commit install

lint:
	cd server && uv run ruff check .

format:
	cd server && uv run ruff format .
	cd server && uv run ruff check --fix .

test:
	cd server && uv run pytest --strict-markers -m "not manual"

pre-commit:
	cd server && uv run pre-commit run --all-files

build:
	cd server && uv build

server:
	cd server && uv run uvicorn faice.main:app --reload

clean:
	rm -rf server/.pytest_cache server/.ruff_cache server/.mypy_cache server/build server/dist server/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
