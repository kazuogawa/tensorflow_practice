run/beginner:
	poetry run python src/main.py

formatter:
	poetry run black .
	poetry run isort .

docker-build:
	docker build -t tensorflow_practice -f Dockerfile .

docker-run:
	docker run tensorflow_practice
