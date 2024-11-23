FROM python:3.9-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libssl-dev \
    libffi-dev \
    python3-dev \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure Poetry
RUN pip install poetry

# Install Python dependencies
RUN poetry cache clear --all . && poetry install --no-cache --without dev

COPY ./mdsist /app/mdsist

CMD ["poetry", "run", "fastapi", "run", "mdsist/app.py", "--port", "80"]
