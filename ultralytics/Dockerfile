# Strip dataset generator (CPU). Build from this folder:
#   docker build -t strip-gen .
FROM python:3.10-slim

ENV PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements-gen.txt .
RUN pip install -r requirements-gen.txt

COPY lib.py strip_pre2.py gen_folds.py ./

# /data: <dataset>/<dataset>.csv (a pca_result.csv is written next to it), /out: results
VOLUME ["/data", "/out"]
ENTRYPOINT ["python", "gen_folds.py"]
