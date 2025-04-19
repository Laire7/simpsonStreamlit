FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Streamlit port
EXPOSE 8501

CMD ["streamlit", "run", "simpson_test.py", "--server.port=8501", "--server.address=0.0.0.0"]

