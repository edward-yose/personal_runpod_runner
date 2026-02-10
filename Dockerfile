FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install vLLM with compatible versions
RUN pip install --no-cache-dir \
    vllm==0.6.4.post1 \
    transformers==4.45.2 \
    accelerate \
    runpod

# Set HF token at build time or runtime
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

COPY handler.py .

CMD ["python", "-u", "handler.py"]