_run/%:
	poetry run python src/main.py ${executor_name}

run/mnist: executor_name := mnist
run/mnist: _run/mnist

run/fashion_mnist: executor_name := fashion_mnist
run/fashion_mnist: _run/fashion_mnist

format:
	poetry run black .
	poetry run isort .

docker-build:
	docker build -t tensorflow_practice -f Dockerfile .

docker-run:
	docker run tensorflow_practice
