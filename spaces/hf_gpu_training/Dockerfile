# HF GPU Space — uses your Hugging Face balance while the Space is on GPU hardware.
#
# Build context = your **Git repository root**.
# - If the repo root contains a `DART/` folder (e.g. `mano/DART/...`), use default DART_PREFIX.
# - If this Space repo **is** the DART root, create a Docker build-arg `DART_PREFIX=.` in Space settings.
#
# Space → Settings → Dev mode → Dockerfile path: `DART/spaces/hf_gpu_training/Dockerfile`

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
ARG DART_PREFIX=DART

WORKDIR /app

COPY ${DART_PREFIX}/requirements.txt ${DART_PREFIX}/requirements_hackathon.txt /app/
COPY ${DART_PREFIX}/env /app/env
COPY ${DART_PREFIX}/reward /app/reward
COPY ${DART_PREFIX}/tools /app/tools
COPY ${DART_PREFIX}/evaluation /app/evaluation
COPY ${DART_PREFIX}/training /app/training
COPY ${DART_PREFIX}/scripts /app/scripts
COPY ${DART_PREFIX}/spaces/hf_gpu_training/entrypoint.sh /app/entrypoint.sh

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements_hackathon.txt

RUN mkdir -p /app/logs /app/docs/figures && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
