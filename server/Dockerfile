ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app/env
COPY . /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \
    pip install uv; \
    fi

RUN uv pip install --system -e .

ENV PYTHONPATH="/app/env:${PYTHONPATH}"
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
