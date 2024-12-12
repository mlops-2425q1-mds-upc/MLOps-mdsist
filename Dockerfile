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

# Copy reference data for evidently
RUN mkdir evidently-monitoring
COPY ./evidently-monitoring/reference_data.csv /app/evidently-monitoring/reference_data.csv

# Copy run.sh file
COPY ./run.sh /app/run.sh
RUN chmod +x /app/run.sh

# COpy codebase
COPY ./mdsist /app/mdsist

# Run
CMD ["./run.sh"]
