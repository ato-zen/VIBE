list:
    @just --list

# Run all the formatting, linting, and testing commands
lint:
    ruff format .
    ruff check . --fix
    ty check .

# remove Python file artifacts
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
