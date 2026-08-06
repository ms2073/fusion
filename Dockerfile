FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /bin/uv

WORKDIR /app

# Dependency layer, cached independently of source changes.
COPY pyproject.toml README.md ./
RUN uv pip install --system --no-cache .

COPY fusion/ ./fusion/
COPY config.yaml ./

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('FUSION_PORT','8787')}/healthz\").read()"

CMD ["python", "-m", "fusion"]
