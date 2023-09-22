format:
	poetry run black src
	poetry run isort src

lint:
	poetry run flake8
	poetry run pyright
