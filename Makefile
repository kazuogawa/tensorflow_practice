_run/%:
	poetry run python src/main.py ${executor_name}

run/minist: executor_name := minist
run/minist: _run/minist

run/fashion_minist: executor_name := fashion_minist
run/fashion_minist: _run/fashion_minist

format:
	poetry run black .
	poetry run isort .

docker-build:
	docker build -t tensorflow_practice -f Dockerfile .

docker-run:
	docker run tensorflow_practice
