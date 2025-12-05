install:
	pip install uv &&\
	uv sync

lint:
	uv run pylint --rcfile=.pylintrc --ignore-patterns=test_.*\.py mylib api cli scripts

format:
	uv run black mylib cli api tests

test:
	uv run python -m pytest tests/ -vv --cov=mylib --cov=api --cov=cli

refactor: format lint

all: install format lint test

