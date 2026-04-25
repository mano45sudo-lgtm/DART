FROM python:3.10-slim

WORKDIR /app

# Copy only what we need for the env server
COPY dtm_openenv /app/dtm_openenv
COPY env /app/env
COPY tools /app/tools
COPY reward /app/reward
# requirements_hackathon.txt includes `-r requirements.txt`; both must be present
COPY requirements.txt /app/requirements.txt
COPY requirements_hackathon.txt /app/requirements_hackathon.txt

RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements_hackathon.txt && \
    pip install --no-cache-dir fastapi uvicorn requests

EXPOSE 7860

# Hugging Face Spaces expects port 7860
CMD ["uvicorn", "dtm_openenv.server.app:app", "--host", "0.0.0.0", "--port", "7860"]

