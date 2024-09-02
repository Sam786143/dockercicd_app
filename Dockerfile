FROM python:3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx

# Copy requirements first to leverage Docker cache
COPY app/requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt
RUN pip install -U langchain-community
RUN apt-get update && apt-get install -y libglib2.0-0

# Copy the rest of the application
COPY app .

# Set the command to run your application
CMD ["streamlit", "run", "main.py", "--server.port=8501"]
