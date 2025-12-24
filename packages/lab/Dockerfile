FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./

COPY packages/packageA ./packages/packageA

RUN uv sync --frozen --no-dev --project packages/packageA

CMD ["uv", "run", "--project", "packages/packageA", "packageA"]
