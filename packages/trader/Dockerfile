FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock conftest.py ./
COPY packages/domain ./packages/domain
COPY packages/store ./packages/store
COPY packages/listener ./packages/listener
COPY packages/modeler ./packages/modeler
COPY packages/trader ./packages/trader

RUN uv sync --frozen --no-dev --project packages/trader

CMD ["uv", "run", "--project", "packages/trader", "python", "-m", "trader.orchestrator"]
