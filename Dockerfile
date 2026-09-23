FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY eval/ ./eval/
COPY scripts/ ./scripts/

# The warehouse is generated from a fixed seed, so building it into the image
# keeps the repository free of a 3.5 MB binary while the container still
# starts with data already in place.
RUN python scripts/build_warehouse.py

# The two real public datasets (UCI Online Retail, Our World in Data CO2),
# downloaded and imported at build time. If a source is unreachable the image
# still builds; the app simply does not offer that dataset.
RUN python scripts/build_examples.py --allow-missing

# Uploaded tables live on their own volume (see docker-compose.yml), so they
# survive rebuilds. The directory must exist in the image: a volume mounted
# over a missing path comes up owned by root and the app could not write it.
RUN mkdir -p /app/userdata     && useradd --create-home --uid 1000 app     && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
