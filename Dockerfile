FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.10.12

# Hugging Face Spaces convention: run as uid 1000 with a writable home
RUN useradd --create-home --uid 1000 user
USER user
WORKDIR /home/user/app
ENV UV_PROJECT_ENVIRONMENT=/home/user/app/.venv \
    PATH=/home/user/app/.venv/bin:$PATH

# Resolve and install runtime dependencies in a cached layer
COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Install the project itself on top of the cached dependency layer
COPY --chown=user src/ ./src/
COPY --chown=user app/ ./app/
COPY --chown=user .streamlit/ ./.streamlit/
RUN uv sync --frozen --no-dev

EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
