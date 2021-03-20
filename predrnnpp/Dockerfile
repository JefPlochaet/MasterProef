FROM tensorflow/tensorflow:latest-gpu

RUN apt update -y && apt upgrade -y && apt install -y \ 
    htop && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN mkdir -p /workspace

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY multiple .

CMD ["bash", "multiple"]
