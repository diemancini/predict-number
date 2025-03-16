FROM python:3.9-slim As builder

# Set the working directory inside the container
WORKDIR /app

# Create a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  libopenblas-dev \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Stage 2: Final stage
FROM python:3.9-slim

# Set the working directory
WORKDIR /backend/src

# Install runtime dependencies for PyTorch
# RUN apt-get update && apt-get install -y --no-install-recommends \
#   libopenblas-dev \
#   && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set the PATH to include the virtual environment's bin directory
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /app /backend/src

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
