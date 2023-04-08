FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install -y \
    python3 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

RUN pip install pyyaml --no-cache-dir \
    pandas \
    tqdm \
    pytest \
    torch-summary \
    scikit-learn

COPY . /workspace
WORKDIR /workspace