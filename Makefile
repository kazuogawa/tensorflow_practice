#_run/%:
#	poetry run {TARGET}

run/beginner:
	poetry run python src/main.py

formatter:
	poetry run black .
	poetry run isort .
