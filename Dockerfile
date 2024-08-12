FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set the working directory 
WORKDIR /app

# Install Python 3.10 and other necessary packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Install pip for Python 3.10
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py

# Copy the current directory into /app
COPY . /app

# Install needed packages
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir torchmetrics pandas wandb opencv-python pycocotools "torchmetrics[detection]"

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
