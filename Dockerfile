FROM python:3.9

RUN pip install poetry

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-cache --without dev

COPY ./mdsist /app/mdsist

CMD ["poetry", "run", "fastapi", "run", "mdsist/app.py", "--port", "80"]
