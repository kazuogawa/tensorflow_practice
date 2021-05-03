FROM python:3.8.0-slim
RUN apt-get update && apt-get install -y curl
RUN python -m pip install --upgrade pip

ENV POETRY_VERSION=1.1.4
ENV POETRY_HOME="/opt/poetry"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

COPY . tensorflow_practice
WORKDIR /tensorflow_practice

RUN poetry install --no-dev

ENTRYPOINT ["bash", "./run.sh"]
