# Start with a Python base image (replace with specific version required)
FROM python:3.11.7-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt to the container before installing dependencies
COPY requirements.txt .

# Install system dependencies, including build-essential and Rust using rustup
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust via rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    export PATH="$HOME/.cargo/bin:$PATH" && \
    pip install --no-cache-dir -r requirements.txt

# Package Update
# Changed: kaleido==0.1.0.post1 -> kaleido==0.1.0
# Changed: mkl_fft==1.3.10 -> mkl_fft==1.3.11
# Changed: mkl_random==1.2.7 -> mkl_random==1.2.8
# Changed: mkl-service==2.4.0 -> mkl-service==2.4.1
# Changed: pywin32==305.1 -> pywin32==305.1; platform_system == "Windows"
# Changed: pywinpty==2.0.10 -> pywinpty; platform_system == "Windows"

# Copy the entire src folder into the container
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Set the default command to run your application (adjust as needed)
CMD ["streamlit", "run", "src/frontend/streamlit_app.py"]

# To build Dockerfile
# docker build -t my-streamlit-app .

# To run built container
# docker run -p 8501:8501 --env-file streamlit.env -v ${PWD}:/app my-streamlit-app