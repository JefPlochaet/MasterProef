FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN apt update -y && apt upgrade -y && apt install -y \ 
    htop && \
    rm -rf /var/lib/apt/lists/*

RUN pip install matplotlib

RUN mkdir -p /data /workspace

CMD ["python3", "/workspace/tuts.py"]
